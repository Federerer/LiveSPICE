using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Reflection;
using System.Reflection.Emit;
using ComputerAlgebra;
using ComputerAlgebra.LinqCompiler;
using Util;
using LinqExprs = System.Linq.Expressions;
using LinqExpr = System.Linq.Expressions.Expression;
using static System.Linq.Expressions.Expression;
using ParamExpr = System.Linq.Expressions.ParameterExpression;
using AgileObjects.ReadableExpressions;
using FastExpressionCompiler;
using ExpressionToCodeLib;
using System.Security.Permissions;
using System.Security;
using System.Collections.Specialized;
using System.Numerics;
using DelegateHostAssembly;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Circuit
{
    /// <summary>
    /// Exception thrown when a simulation does not converge.
    /// </summary>
    public class SimulationDiverged : FailedToConvergeException
    {
        private long at;
        /// <summary>
        /// Sample number at which the simulation diverged.
        /// </summary>
        public long At { get { return at; } }

        public SimulationDiverged(string Message, long At) : base(Message) { at = At; }

        public SimulationDiverged(int At) : base("Simulation diverged.") { at = At; }
    }

    /// <summary>
    /// Simulate a circuit.
    /// </summary>
    public class Simulation
    {

        protected static readonly Variable t = TransientSolution.t;
        protected Expression t0 { get { return t - Solution.TimeStep; } }
        protected Arrow t_t0 { get { return Arrow.New(t, t0); } }

        private long _n = 0;
        /// <summary>
        /// Get which sample the simulation is at.
        /// </summary>
        public long At { get { return _n; } }
        /// <summary>
        /// Get the simulation time.
        /// </summary>
        public double Time { get { return At * TimeStep; } }

        /// <summary>
        /// Get the timestep for the simulation.
        /// </summary>
        public double TimeStep { get { return (double)(Solution.TimeStep * _oversample); } }

        private ILog log = new NullLog();
        /// <summary>
        /// Log associated with this simulation.
        /// </summary>
        public ILog Log { get { return log; } set { log = value; } }

        private TransientSolution solution;
        private readonly bool _vectorize;

        /// <summary>
        /// Solution of the circuit we are simulating.
        /// </summary>
        public TransientSolution Solution
        {
            get { return solution; }
            set { solution = value; InvalidateProcess(); }
        }

        private int _oversample = 8;
        /// <summary>
        /// Oversampling factor for this simulation.
        /// </summary>
        public int Oversample { get { return _oversample; } set { _oversample = value; InvalidateProcess(); } }

        private int _iterations = 8;
        /// <summary>
        /// Maximum number of iterations allowed for the simulation to converge.
        /// </summary>
        public int Iterations { get { return _iterations; } set { _iterations = value; InvalidateProcess(); } }

        /// <summary>
        /// The sampling rate of this simulation, the sampling rate of the transient solution divided by the oversampling factor.
        /// </summary>
        public Expression SampleRate { get { return 1 / (Solution.TimeStep * _oversample); } }

        private Expression[] _inputs = new Expression[] { };
        /// <summary>
        /// Expressions representing input samples.
        /// </summary>
        public Expression[] Inputs { get { return _inputs; } set { _inputs = value; InvalidateProcess(); } }

        private Expression[] _outputs = new Expression[] { };
        /// <summary>
        /// Expressions for output samples.
        /// </summary>
        public Expression[] Outputs { get { return _outputs; } set { _outputs = value; InvalidateProcess(); } }

        // Stores any global state in the simulation (previous state values, mostly).
        private readonly Dictionary<Expression, GlobalExpr<double>> _globals = new Dictionary<Expression, GlobalExpr<double>>();

        // Add a new global and set it to 0 if it didn't already exist.
        private void AddGlobal(Expression Name)
        {
            if (!_globals.ContainsKey(Name))
                _globals.Add(Name, new GlobalExpr<double>(0.0));
        }


        private double[] _state;
        private double[] _initialState;

        /// <summary>
        /// Create a simulation using the given solution and the specified inputs/outputs.
        /// </summary>
        /// <param name="Solution">Transient solution to run.</param>
        /// <param name="Input">Expressions in the solution to be defined by input samples.</param>
        /// <param name="Output">Expressions describing outputs to be saved from the simulation.</param>
        public Simulation(TransientSolution Solution, Expression[] inputs, Expression[] outputs, int oversample, int iterations)
        {
            solution = Solution;
            _inputs = inputs;
            _outputs = outputs;
            _oversample = oversample;
            _iterations = iterations;


            // If any system depends on the previous value of an unknown, we need a global variable for it.
            foreach (Expression i in Solution.Solutions.SelectMany(i => i.Unknowns))
            {
                if (Solution.Solutions.Any(j => j.DependsOn(i.Evaluate(t, t0))))
                {
                    AddGlobal(i.Evaluate(t, t0));
                }
            }

            // Also need globals for any Newton's method unknowns.
            foreach (Expression i in Solution.Solutions.OfType<NewtonIteration>().SelectMany(i => i.Unknowns))
                AddGlobal(i.Evaluate(t, t0));

            // Set the global values to the initial conditions of the solution.
            foreach (KeyValuePair<Expression, GlobalExpr<double>> i in _globals)
            {
                Expression init = i.Key.Evaluate(t0, 0).Evaluate(Solution.InitialConditions);
                i.Value.Value = init is Constant ? (double)init : 0.0;
            }

            InvalidateProcess();
        }

        /// <summary>
        /// Process some samples with this simulation. The Input and Output buffers must match the enumerations provided
        /// at initialization.
        /// </summary>
        /// <param name="N">Number of samples to process.</param>
        /// <param name="Input">Buffers that describe the input samples.</param>
        /// <param name="Output">Buffers to receive output samples.</param>
        public void Run(int N, IEnumerable<double[]> Input, IEnumerable<double[]> Output)
        {
            if (process == null)
                process = DefineProcess();

            try
            {
                try
                {
                    process(N, _n * TimeStep, _state, Input.ToArray(), Output.ToArray());
                    _n += N;
                }
                catch (TargetInvocationException Ex)
                {
                    throw Ex.InnerException;
                }
            }
            catch (SimulationDiverged Ex)
            {
                throw new SimulationDiverged("Simulation diverged near t = " + Quantity.ToString(Time, Units.s) + " + " + Ex.At, _n + Ex.At);
            }
        }
        public void Run(int N, IEnumerable<double[]> Output) { Run(N, new double[][] { }, Output); }
        public void Run(double[] Input, IEnumerable<double[]> Output) { Run(Input.Length, new[] { Input }, Output); }
        public void Run(double[] Input, double[] Output) { Run(Input.Length, new[] { Input }, new[] { Output }); }

        private Action<int, double, double[], double[][], double[][]> process = null;
        // Rebuild the process function.
        private void InvalidateProcess()
        {
            try
            {
                process = null;
                process = DefineProcess();
            }
            catch (Exception) { }
        }

        // The resulting lambda processes N samples, using buffers provided for Input and Output:
        //  void Process(int N, double t0, double T, double[] Input0 ..., double[] Output0 ...)
        //  { ... }
        private Action<int, double, double[], double[][], double[][]> DefineProcess()
        {
            // Map expressions to identifiers in the syntax tree.
            var inputs = new List<(Expression Expression, LinqExpr LinqExpression)>();
            var outputs = new List<(Expression Expression, LinqExpr LinqExpression)>();

            // Lambda code generator.
            CodeGen code = new CodeGen();

            // Create parameters for the basic simulation info (N, t, Iterations).
            ParamExpr SampleCount = code.Decl<int>(Scope.Parameter, "SampleCount");
            ParamExpr t = code.Decl(Scope.Parameter, Simulation.t);

            //global state as array
            var state = code.Decl<double[]>(Scope.Parameter, "state");

            var ins = code.Decl<double[][]>(Scope.Parameter, "ins");
            var outs = code.Decl<double[][]>(Scope.Parameter, "outs");

            // Create buffer parameters for each input...
            for (int i = 0; i < Inputs.Length; i++)
            {
                inputs.Add((Inputs[i], ArrayAccess(ins, Constant(i))));
            }

            // ... and output.
            for (int i = 0; i < Outputs.Length; i++)
            {
                outputs.Add((Outputs[i], ArrayAccess(outs, Constant(i))));
            }

            // Create globals to store previous values of inputs.
            foreach (Expression i in Inputs.Distinct())
            {
                AddGlobal(i.Evaluate(t_t0));
            }

            _state = _globals.Values.Select(g => g.Value).ToArray();
            _initialState = _state;

            // Define lambda body.

            // int Zero = 0
            var zero = Constant(0);

            // double h = T / Oversample
            var h = Constant(TimeStep / (double)Oversample);


            // Load the globals to local variables and add them to the map.
            foreach (var (item, index) in _globals.Select((item, index) => (item, index)))
            {
                code[item.Key] = ArrayAccess(state, Constant(index));
            }

            foreach (var input in inputs)
                code.Add(Assign(code.Decl(input.Expression), code[input.Expression.Evaluate(t_t0)]));

            // Create arrays for linear systems.
            int M = Solution.Solutions.OfType<NewtonIteration>().Max(i => i.Equations.Count(), 0);
            int N = Solution.Solutions.OfType<NewtonIteration>().Max(i => i.UnknownDeltas.Count(), 0) + 1;
            LinqExpr JxF = code.DeclInit<double[][]>("JxF", NewArrayBounds(typeof(double[]), Constant(M)));
            for (int j = 0; j < M; ++j)
                code.Add(Assign(ArrayAccess(JxF, Constant(j)), NewArrayBounds(typeof(double), Constant(N))));

            // for (int n = 0; n < SampleCount; ++n)
            ParamExpr n = code.Decl<int>("n");
            code.For(
                () => code.Add(Assign(n, zero)),
                LessThan(n, SampleCount),
                () => code.Add(PreIncrementAssign(n)),
                () =>
                {
                    // Prepare input samples for oversampling interpolation.
                    Dictionary<Expression, LinqExpr> dVi = new Dictionary<Expression, LinqExpr>();
                    foreach (Expression i in Inputs.Distinct())
                    {
                        LinqExpr Va = code[i];
                        // Sum all inputs with this key.
                        IEnumerable<LinqExpr> Vbs = inputs.Where(j => j.Expression.Equals(i)).Select(j => j.LinqExpression);
                        LinqExpr Vb = ArrayAccess(Vbs.First(), n);
                        foreach (LinqExpr j in Vbs.Skip(1))
                            Vb = Add(Vb, ArrayAccess(j, n));

                        // dVi = (Vb - Va) / Oversample
                        code.Add(Assign(
                            Decl<double>(code, dVi, i, "d" + i.ToString().Replace("[t]", "")),
                            Multiply(Subtract(Vb, Va), Constant(1.0 / (double)Oversample))));
                    }

                    // Prepare output sample accumulators for low pass filtering.
                    Dictionary<Expression, LinqExpr> Vo = new Dictionary<Expression, LinqExpr>();
                    foreach (Expression i in Outputs.Distinct())
                        code.Add(Assign(
                            Decl<double>(code, Vo, i, i.ToString().Replace("[t]", "")),
                            Constant(0.0)));

                    // int ov = Oversample; 
                    // do { -- ov; } while(ov > 0)
                    ParamExpr ov = code.Decl<int>("ov");
                    code.Add(Assign(ov, Constant(Oversample)));
                    code.DoWhile(() =>
                    {
                        // t += h
                        code.Add(AddAssign(t, h));

                        // Interpolate the input samples.
                        foreach (Expression i in Inputs.Distinct())
                            code.Add(AddAssign(code[i], dVi[i]));

                        // Compile all of the SolutionSets in the solution.
                        foreach (var solutionSet in Solution.Solutions)
                        {
                            if (solutionSet is LinearSolutions linearSolutions)
                            {
                                // Linear solutions are easy.
                                foreach (var arrow in linearSolutions.Solutions)
                                {
                                    code.DeclInit(arrow.Left, arrow.Right);
                                }
                            }
                            else if (solutionSet is NewtonIteration newtonIteration)
                            {
                                // Start with the initial guesses from the solution.
                                foreach (Arrow i in newtonIteration.Guesses)
                                {
                                    code.DeclInit(i.Left, i.Right);
                                }

                                // int it = iterations
                                LinqExpr it = code.ReDeclInit("it", Iterations);
                                // do { ... --it } while(it > 0)
                                code.DoWhile((Break) =>
                                {
                                    // Solve the un-solved system.
                                    Solve(code, JxF, newtonIteration.Equations, newtonIteration.UnknownDeltas);

                                    // Compile the pre-solved solutions.
                                    foreach (Arrow i in newtonIteration.KnownDeltas ?? Enumerable.Empty<Arrow>())
                                    {
                                        code.DeclInit(i.Left, i.Right);
                                    }

                                    // bool done = true
                                    LinqExpr done = code.ReDeclInit("done", true);
                                    foreach (Expression i in newtonIteration.Unknowns)
                                    {
                                        LinqExpr v = code[i];
                                        LinqExpr dv = code[NewtonIteration.Delta(i)];

                                        // done &&= (|dv| < |v|*epsilon)
                                        code.Add(Assign(done, AndAlso(done, LessThan(Multiply(Abs(dv), Constant(1e4)), Add(Abs(v), Constant(1e-6))))));
                                        // v += dv
                                        code.Add(AddAssign(v, dv));
                                    }
                                    // if (done) break
                                    code.Add(IfThen(done, Break));

                                    // --it;
                                    code.Add(PreDecrementAssign(it));
                                }, GreaterThan(it, zero));

                                //// bool failed = false
                                //LinqExpr failed = Decl(code, code, "failed", LinqExpr.Constant(false));
                                //for (int i = 0; i < eqs.Length; ++i)
                                //    // failed |= |JxFi| > epsilon
                                //    code.Add(LinqExpr.OrAssign(failed, LinqExpr.GreaterThan(Abs(eqs[i].ToExpression().Compile(map)), LinqExpr.Constant(1e-3))));

                                //code.Add(LinqExpr.IfThen(failed, ThrowSimulationDiverged(n)));
                            }
                        }

                        // Update the previous timestep variables.
                        foreach (SolutionSet S in Solution.Solutions)
                            foreach (Expression unknown in S.Unknowns.Where(i => _globals.Keys.Contains(i.Evaluate(t_t0))))
                                code.Add(Assign(code[unknown.Evaluate(t_t0)], code[unknown]));

                        // Vo += i
                        foreach (Expression i in Outputs.Distinct())
                        {
                            LinqExpr Voi = Constant(0.0);
                            try
                            {
                                Voi = code.Compile(i);
                            }
                            catch (Exception Ex)
                            {
                                Log.WriteLine(MessageType.Warning, Ex.Message);
                            }
                            code.Add(AddAssign(Vo[i], Voi));
                        }

                        // Vi_t0 = Vi
                        foreach (Expression i in Inputs.Distinct())
                            code.Add(Assign(code[i.Evaluate(t_t0)], code[i]));

                        // --ov;
                        code.Add(PreDecrementAssign(ov));
                    }, GreaterThan(ov, zero));

                    // Output[i][n] = Vo / Oversample
                    foreach (var output in outputs)
                        code.Add(Assign(ArrayAccess(output.LinqExpression, n), Multiply(Vo[output.Expression], Constant(1.0 / Oversample))));

                    // Every 256 samples, check for divergence.
                    if (Vo.Count > 0)
                    {
                        code.Add(IfThen(Equal(And(n, Constant(0xFF)), zero),
                            Block(Vo.Select(i => IfThenElse(IsNotReal(i.Value),
                                ThrowSimulationDiverged(n),
                                Assign(i.Value, RoundDenormToZero(i.Value)))))));
                    }
                });

            // Copy the global state variables back to the globals.
            //foreach (var (item, index) in _globals.Select((item, index) => (item, index)))
            //{
            //    code.Add(Assign(ArrayAccess(state, Constant(index)), code[item.Key]));
            //}
            // code.Add(_globals.Select(g => Assign(g.Value.Expression, code[g.Key])).ToArray());

            var lambda = code.Build<Action<int, double, double[], double[][], double[][]>>();

            var str = lambda.ToReadableString();
#if NET472
            return GetCompiledDelegate(lambda);
#else
            return lambda.Compile();
#endif

            //        var da = AppDomain.CurrentDomain.DefineDynamicAssembly(
            //new AssemblyName("dyn"), // call it whatever you want
            //AssemblyBuilderAccess.Save);

            //        var dm = da.DefineDynamicModule("dyn_mod", "dyn.dll");
            //        var dt = dm.DefineType("dyn_type");
            //        var method = dt.DefineMethod(
            //            "Foo",
            //            MethodAttributes.Public | MethodAttributes.Static);

            //        lambda.CompileToMethod(method);
            //        dt.CreateType();

            //        da.Save("dyn.dll");



        }

#if NET472
        public static T GetCompiledDelegate<T>(LinqExprs.Expression<T> expr)
        {
            var assemblyName = new AssemblyName("DelegateHostAssembly") { Version = new Version("1.0.0.0") };

            var assemblyBuilder =
                AppDomain.CurrentDomain.DefineDynamicAssembly(
                    assemblyName,
                    AssemblyBuilderAccess.RunAndSave);
            var moduleBuilder = assemblyBuilder.DefineDynamicModule("DelegateHostAssembly", "DelegateHostAssembly.dll");
            var typeBuilder = moduleBuilder.DefineType("DelegateHostAssembly." + "foo", TypeAttributes.Public);
            var methBldr = typeBuilder.DefineMethod("Execute", MethodAttributes.Public | MethodAttributes.Static);

            expr.CompileToMethod(methBldr);
            Type myType = typeBuilder.CreateType();

            assemblyBuilder.Save("DelegateHostAssembly.dll");


            var mi = myType.GetMethod("Execute");

            // have to box to object because .NET doesn't allow Delegates as generic constraints,
            // nor does it allow casting of Delegates to generic type variables like "T"
            object foo = Delegate.CreateDelegate(typeof(T), mi);

            return (T)foo;
        }
#endif

        // Solve a system of linear equations
        private void Solve(CodeGen code, LinqExpr Ab, IEnumerable<LinearCombination> Equations, IEnumerable<Expression> Unknowns)
        {
            LinearCombination[] eqs = Equations.ToArray();
            Expression[] deltas = Unknowns.ToArray();

            int M = eqs.Length;
            int N = deltas.Length;

            // Initialize the matrix.
            for (int i = 0; i < M; ++i)
            {
                LinqExpr Abi = code.ReDeclInit<double[]>("Abi", ArrayAccess(Ab, Constant(i)));
                for (int x = 0; x < N; ++x)
                {
                    code.Add(Assign(
                        ArrayAccess(Abi, Constant(x)),
                        code.Compile(eqs[i][deltas[x]])));
                }

                code.Add(Assign(
                    ArrayAccess(Abi, Constant(N)),
                    code.Compile(eqs[i][1])));
            }

            // Gaussian elimination on this turd.
            //RowReduce(code, Ab, M, N);
            var method = _vectorize ? "RowReduceVect" : "RowReduce";
            code.Add(Call(
                GetMethod<Simulation>(method, Ab.Type, typeof(int), typeof(int)),
                Ab,
                Constant(M),
                Constant(N)));

            // Ab is now upper triangular, solve it.
            for (int j = N - 1; j >= 0; --j)
            {
                LinqExpr Abj = code.ReDeclInit<double[]>("Abj", ArrayAccess(Ab, Constant(j)));

                LinqExpr r = ArrayAccess(Abj, Constant(N));
                for (int ji = j + 1; ji < N; ++ji)
                    r = Add(r, Multiply(ArrayAccess(Abj, Constant(ji)), code[deltas[ji]]));
                code.DeclInit(deltas[j], Divide(Negate(r), ArrayAccess(Abj, Constant(j))));
            }
        }

        // A human readable implementation of RowReduce.
        public static void RowReduce(double[][] Ab, int M, int N)
        {
            // Solve for dx.
            // For each variable in the system...
            for (int j = 0; j + 1 < N; ++j)
            {
                int pi = j;
                double max = Math.Abs(Ab[j][j]);

                // Find a pivot row for this variable.
                for (int i = j + 1; i < M; ++i)
                {
                    // if(|JxF[i][j]| > max) { pi = i, max = |JxF[i][j]| }
                    double maxj = Math.Abs(Ab[i][j]);
                    if (maxj > max)
                    {
                        pi = i;
                        max = maxj;
                    }
                }

                // Swap pivot row with the current row.
                if (pi != j)
                {
                    var tmp = Ab[pi];
                    Ab[pi] = Ab[j];
                    Ab[j] = tmp;
                }

                // Eliminate the rows after the pivot.
                double p = Ab[j][j];
                for (int i = j + 1; i < M; ++i)
                {
                    double s = Ab[i][j] / p;
                    if (s != 0.0d)
                    {
                        for (int jj = j + 1; jj <= N; ++jj)
                        {
                            Ab[i][jj] -= Ab[j][jj] * s;
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RowReduceVect(double[][] Ab, int M, int N)
        {
            const double tiny = 0.00001;

            // Solve for dx.
            // For each variable in the system...
            for (int j = 0; j + 1 < N; ++j)
            {
                int pi = j;
                double max = Math.Abs(Ab[j][j]);

                // Find a pivot row for this variable.
                for (int i = j + 1; i < M; ++i)
                {
                    // if(|JxF[i][j]| > max) { pi = i, max = |JxF[i][j]| }
                    double maxj = Math.Abs(Ab[i][j]);
                    if (maxj > max)
                    {
                        pi = i;
                        max = maxj;
                    }
                }

                // Swap pivot row with the current row.
                if (pi != j)
                {
                    var tmp = Ab[pi];
                    Ab[pi] = Ab[j];
                    Ab[j] = tmp;
                }

                var vectorLength = Vector<double>.Count;
                // Eliminate the rows after the pivot.
                double p = Ab[j][j];
                for (int i = j + 1; i < M; ++i)
                {
                    double s = Ab[i][j] / p;
                    if (Math.Abs(s) >= tiny)
                    {
                        int jj;
                        for (jj = j + 1; jj <= (N - vectorLength); jj += vectorLength)
                        {
                            var source = new Vector<double>(Ab[j], jj);
                            var target = new Vector<double>(Ab[i], jj);
                            var res = target - (source * s);
                            res.CopyTo(Ab[i], jj);
                        }
                        for (; jj <= N; ++jj)
                        {
                            Ab[i][jj] -= Ab[j][jj] * s;
                        }
                    }
                }
            }
        }

        // Generate code to perform row reduction.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void RowReduce(CodeGen code, LinqExpr Ab, int M, int N)
        {
            // For each variable in the system...
            for (int j = 0; j + 1 < N; ++j)
            {
                LinqExpr Abj = code.ReDeclInit<double[]>("Abj", ArrayAccess(Ab, Constant(j)));
                // int pi = j
                LinqExpr pi = code.ReDeclInit<int>("pi", Constant(j));
                // double max = |Ab[j][j]|
                LinqExpr max = code.ReDeclInit<double>("max", Abs(ArrayAccess(Abj, Constant(j))));

                // Find a pivot row for this variable.
                //code.For(j + 1, M, _i =>
                //{
                for (int i = j + 1; i < M; ++i)
                {
                    LinqExpr _i = Constant(i);

                    // if(|Ab[i][j]| > max) { pi = i, max = |Ab[i][j]| }
                    LinqExpr maxj = code.ReDeclInit<double>("maxj", Abs(ArrayAccess(ArrayAccess(Ab, _i), Constant(j))));
                    code.Add(IfThen(
                        GreaterThan(maxj, max),
                        Block(
                            Assign(pi, _i),
                            Assign(max, maxj))));
                }

                // (Maybe) swap the pivot row with the current row.
                LinqExpr Abpi = code.ReDecl<double[]>("Abpi");
                code.Add(IfThen(
                    NotEqual(Constant(j), pi), Block(
                        new[] { Assign(Abpi, ArrayAccess(Ab, pi)) }.Concat(
                        Enumerable.Range(j, N + 1 - j).Select(x => Swap(
                            ArrayAccess(Abj, Constant(x)),
                            ArrayAccess(Abpi, Constant(x)),
                            code.ReDecl<double>("swap")))))));

                //// It's hard to believe this swap isn't faster than the above...
                //code.Add(LinqExpr.IfThen(LinqExpr.NotEqual(_j, pi), LinqExpr.Block(
                //    Swap(LinqExpr.ArrayAccess(Ab, _j), LinqExpr.ArrayAccess(Ab, pi), Redeclare<double[]>(code, "temp")),
                //    LinqExpr.Assign(Abj, LinqExpr.ArrayAccess(Ab, _j)))));

                // Eliminate the rows after the pivot.
                LinqExpr p = code.ReDeclInit<double>("p", ArrayAccess(Abj, Constant(j)));
                //code.For(j + 1, M, _i =>
                //{
                for (int i = j + 1; i < M; ++i)
                {
                    LinqExpr Abi = code.ReDeclInit<double[]>("Abi", ArrayAccess(Ab, Constant(i)));

                    // s = Ab[i][j] / p
                    LinqExpr s = code.ReDeclInit<double>("scale", Divide(ArrayAccess(Abi, Constant(j)), p));
                    // Ab[i] -= Ab[j] * s
                    for (int ji = j + 1; ji < N + 1; ++ji)
                        code.Add(SubtractAssign(
                            ArrayAccess(Abi, Constant(ji)),
                            Multiply(ArrayAccess(Abj, Constant(ji)), s)));
                }
            }
        }

        // Returns a throw SimulationDiverged expression at At.
        private LinqExpr ThrowSimulationDiverged(LinqExpr At)
        {
            return Throw(New(typeof(SimulationDiverged).GetConstructor(new Type[] { At.Type }), At));
        }

        private static ParamExpr Decl<T>(CodeGen Target, ICollection<KeyValuePair<Expression, LinqExpr>> Map, Expression Expr, string Name)
        {
            ParamExpr p = Target.Decl<T>(Name);
            Map.Add(new KeyValuePair<Expression, LinqExpr>(Expr, p));
            return p;
        }

        private static ParamExpr Decl<T>(CodeGen Target, ICollection<KeyValuePair<Expression, LinqExpr>> Map, Expression Expr)
        {
            return Decl<T>(Target, Map, Expr, Expr.ToString());
        }

        private static LinqExpr ConstantExpr(double x, Type T)
        {
            if (T == typeof(double))
                return Constant(x);
            else if (T == typeof(float))
                return Constant((float)x);
            else
                throw new NotImplementedException("Constant");
        }

        private static void Swap(ref double a, ref double b) { double t = a; a = b; b = t; }

        // Get a method of T with the given name/param types.
        private static MethodInfo GetMethod(Type T, string Name, params Type[] ParamTypes) { return T.GetMethod(Name, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic, null, ParamTypes, null); }
        private static MethodInfo GetMethod<T>(string Name, params Type[] ParamTypes) { return GetMethod(typeof(T), Name, ParamTypes); }

        // Returns 1 / x.
        private static LinqExpr Reciprocal(LinqExpr x) { return Divide(ConstantExpr(1.0, x.Type), x); }
        // Returns abs(x).
        private static LinqExpr Abs(LinqExpr x) { return Call(GetMethod(typeof(Math), "Abs", x.Type), x); }
        // Returns x*x.
        private static LinqExpr Square(LinqExpr x) { return Multiply(x, x); }

        // Returns true if x is not NaN or Inf
        private static LinqExpr IsNotReal(LinqExpr x)
        {
            return Or(
                Call(GetMethod(x.Type, "IsNaN", x.Type), x),
                Call(GetMethod(x.Type, "IsInfinity", x.Type), x));
        }
        // Round x to zero if it is sub-normal.
        private static LinqExpr RoundDenormToZero(LinqExpr x) { return x; }
        // Generate expression to swap a and b, using t as a temporary.
        private static LinqExpr Swap(LinqExpr a, LinqExpr b, LinqExpr t)
        {
            return Block(
                Assign(t, a),
                Assign(a, b),
                Assign(b, t));
        }
    }
}
