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

        private long n = 0;
        /// <summary>
        /// Get which sample the simulation is at.
        /// </summary>
        public long At { get { return n; } }
        /// <summary>
        /// Get the simulation time.
        /// </summary>
        public double Time { get { return At * TimeStep; } }

        /// <summary>
        /// Get the timestep for the simulation.
        /// </summary>
        public double TimeStep { get { return (double)(Solution.TimeStep * oversample); } }

        private ILog log = new NullLog();
        /// <summary>
        /// Log associated with this simulation.
        /// </summary>
        public ILog Log { get { return log; } set { log = value; } }

        private TransientSolution solution;
        /// <summary>
        /// Solution of the circuit we are simulating.
        /// </summary>
        public TransientSolution Solution 
        { 
            get { return solution; }
            set { solution = value; InvalidateProcess(); }
        }

        private int oversample = 8;
        /// <summary>
        /// Oversampling factor for this simulation.
        /// </summary>
        public int Oversample { get { return oversample; } set { oversample = value; InvalidateProcess(); } }

        private int iterations = 8;
        /// <summary>
        /// Maximum number of iterations allowed for the simulation to converge.
        /// </summary>
        public int Iterations { get { return iterations; } set { iterations = value; InvalidateProcess(); } }

        /// <summary>
        /// The sampling rate of this simulation, the sampling rate of the transient solution divided by the oversampling factor.
        /// </summary>
        public Expression SampleRate { get { return 1 / (Solution.TimeStep * oversample); } }

        private Expression[] input = new Expression[] { };
        /// <summary>
        /// Expressions representing input samples.
        /// </summary>
        public Expression[] Input { get { return input; } set { input = value; InvalidateProcess(); } }

        private Expression[] output = new Expression[] { };
        /// <summary>
        /// Expressions for output samples.
        /// </summary>
        public Expression[] Output { get { return output; } set { output = value; InvalidateProcess(); } }
        
        // Stores any global state in the simulation (previous state values, mostly).
        private Dictionary<Expression, GlobalExpr<double>> globals = new Dictionary<Expression, GlobalExpr<double>>();
        // Add a new global and set it to 0 if it didn't already exist.
        private void AddGlobal(Expression Name)
        {
            if (!globals.ContainsKey(Name))
                globals.Add(Name, new GlobalExpr<double>(0.0));
        }

        /// <summary>
        /// Create a simulation using the given solution and the specified inputs/outputs.
        /// </summary>
        /// <param name="Solution">Transient solution to run.</param>
        /// <param name="Input">Expressions in the solution to be defined by input samples.</param>
        /// <param name="Output">Expressions describing outputs to be saved from the simulation.</param>
        public Simulation(TransientSolution Solution)
        {
            solution = Solution;
            
            // If any system depends on the previous value of an unknown, we need a global variable for it.
            foreach (Expression i in Solution.Solutions.SelectMany(i => i.Unknowns))
                if (Solution.Solutions.Any(j => j.DependsOn(i.Evaluate(t, t0))))
                    AddGlobal(i.Evaluate(t, t0));
            // Also need globals for any Newton's method unknowns.
            foreach (Expression i in Solution.Solutions.OfType<NewtonIteration>().SelectMany(i => i.Unknowns))
                AddGlobal(i.Evaluate(t, t0));

            // Set the global values to the initial conditions of the solution.
            foreach (KeyValuePair<Expression, GlobalExpr<double>> i in globals)
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

            // Build parameter list for the processor.
            object[] parameters = new object[2 + Input.Count() + Output.Count()];
            int p = 0;

            try
            {
                try
                {
                    process(N, n*TimeStep, Input.ToArray(), Output.ToArray());
                    n += N;
                }
                catch (TargetInvocationException Ex)
                {
                    throw Ex.InnerException;
                }
            }
            catch (SimulationDiverged Ex)
            {
                throw new SimulationDiverged("Simulation diverged near t = " + Quantity.ToString(Time, Units.s) + " + " + Ex.At, n + Ex.At);
            }
        }
        public void Run(int N, IEnumerable<double[]> Output) { Run(N, new double[][] { }, Output); }
        public void Run(double[] Input, IEnumerable<double[]> Output) { Run(Input.Length, new[] { Input }, Output); }
        public void Run(double[] Input, double[] Output) { Run(Input.Length, new[] { Input }, new[] { Output }); }

        private Func<int, double, double[][], double[][], double> process = null;
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
        private Func<int, double, double[][], double[][], double> DefineProcess()
        {
            // Map expressions to identifiers in the syntax tree.
            Dictionary<Expression, LinqExpr> inputs = new Dictionary<Expression, LinqExpr>();
            Dictionary<Expression, LinqExpr> outputs = new Dictionary<Expression, LinqExpr>();

            // Lambda code generator.
            CodeGen code = new CodeGen();

            // Create parameters for the basic simulation info (N, t, Iterations).
            ParamExpr SampleCount = code.Decl<int>(Scope.Parameter, "SampleCount");
            ParamExpr t = code.Decl(Scope.Parameter, Simulation.t);

            var ins = code.Decl<double[][]>(Scope.Parameter, "ins");
            var outs = code.Decl<double[][]>(Scope.Parameter, "outs");

            // Create buffer parameters for each input...
            for (int i = 0; i < Input.Length; i++)
            {
                inputs.Add(Input[i], ArrayAccess(ins, Constant(i)));
            }

            // ... and output.
            for (int i = 0; i < Output.Length; i++)
            {
                outputs.Add(Output[i], ArrayAccess(outs, Constant(i)));
            }

            // Create globals to store previous values of inputs.
            foreach (Expression i in Input.Distinct())
                AddGlobal(i.Evaluate(t_t0));

            // Define lambda body.

            // int Zero = 0
            var zero = Constant(0);

            // double h = T / Oversample
            var h = Constant(TimeStep / (double)Oversample);

            // Load the globals to local variables and add them to the map.
            foreach (KeyValuePair<Expression, GlobalExpr<double>> i in globals)
                code.Add(Assign(code.Decl(i.Key), i.Value));

            foreach (KeyValuePair<Expression, LinqExpr> i in inputs)
                code.Add(Assign(code.Decl(i.Key), code[i.Key.Evaluate(t_t0)]));

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
                    foreach (Expression i in Input.Distinct())
                    {
                        LinqExpr Va = code[i];
                        // Sum all inputs with this key.
                        IEnumerable<LinqExpr> Vbs = inputs.Where(j => j.Key.Equals(i)).Select(j => j.Value);
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
                    foreach (Expression i in Output.Distinct())
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
                        foreach (Expression i in Input.Distinct())
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
                                    code.DeclInit(i.Left, i.Right);

                                // int it = iterations
                                LinqExpr it = code.ReDeclInit<int>("it", Iterations);
                                // do { ... --it } while(it > 0)
                                code.DoWhile((Break) =>
                                {
                                    // Solve the un-solved system.
                                    Solve(code, JxF, newtonIteration.Equations, newtonIteration.UnknownDeltas);

                                    // Compile the pre-solved solutions.
                                    if (newtonIteration.KnownDeltas != null)
                                        foreach (Arrow i in newtonIteration.KnownDeltas)
                                            code.DeclInit(i.Left, i.Right);

                                    // bool done = true
                                    LinqExpr done = code.ReDeclInit("done", true);
                                    foreach (Expression i in newtonIteration.Unknowns)
                                    {
                                        LinqExpr v = code[i];
                                        LinqExpr dv = code[NewtonIteration.Delta(i)];

                                        // done &= (|dv| < |v|*epsilon)
                                        code.Add(AndAssign(done, LessThan(Multiply(Abs(dv), Constant(1e4)), Add(Abs(v), Constant(1e-6)))));
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
                            foreach (Expression unknown in S.Unknowns.Where(i => globals.Keys.Contains(i.Evaluate(t_t0))))
                                code.Add(Assign(code[unknown.Evaluate(t_t0)], code[unknown]));

                        // Vo += i
                        foreach (Expression i in Output.Distinct())
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
                        foreach (Expression i in Input.Distinct())
                            code.Add(Assign(code[i.Evaluate(t_t0)], code[i]));

                        // --ov;
                        code.Add(PreDecrementAssign(ov));
                    }, GreaterThan(ov, zero));

                    // Output[i][n] = Vo / Oversample
                    foreach (KeyValuePair<Expression, LinqExpr> i in outputs)
                        code.Add(Assign(ArrayAccess(i.Value, n), Multiply(Vo[i.Key], Constant(1.0 / (double)Oversample))));

                    // Every 256 samples, check for divergence.
                    if (Vo.Any())
                    {
                        code.Add(IfThen(Equal(And(n, Constant(0xFF)), zero),
                            Block(Vo.Select(i => IfThenElse(IsNotReal(i.Value),
                                ThrowSimulationDiverged(n),
                                Assign(i.Value, RoundDenormToZero(i.Value)))))));
                    }
                });

            // Copy the global state variables back to the globals.
            code.Add(globals.Select(g => Assign(g.Value, code[g.Key])).ToArray());

            var lambda = code.Build<Func<int, double, double[][], double[][], double>>();

            var str = lambda.ToReadableString();


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



            return lambda.Compile();
        }

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

            var mi = myType.GetMethod("Execute");

            // have to box to object because .NET doesn't allow Delegates as generic constraints,
            // nor does it allow casting of Delegates to generic type variables like "T"
            object foo = Delegate.CreateDelegate(typeof(T), mi);

            return (T)foo;
        }

        // Solve a system of linear equations
        private static void Solve(CodeGen code, LinqExpr Ab, IEnumerable<LinearCombination> Equations, IEnumerable<Expression> Unknowns)
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
                    code.Add(Assign(
                        ArrayAccess(Abi, Constant(x)),
                        code.Compile(eqs[i][deltas[x]])));
                code.Add(Assign(
                    ArrayAccess(Abi, Constant(N)),
                    code.Compile(eqs[i][1])));
            }

            // Gaussian elimination on this turd.
            //RowReduce(code, Ab, M, N);
            code.Add(Call(
                GetMethod<Simulation>("RowReduce", Ab.Type, typeof(int), typeof(int)),
                Ab,
                Constant(M),
                Constant(N)));

            // Ab is now upper triangular, solve it.
            for (int j = N - 1; j >= 0; --j)
            {
                LinqExpr _j = Constant(j);
                LinqExpr Abj = code.ReDeclInit<double[]>("Abj", ArrayAccess(Ab, _j));

                LinqExpr r = ArrayAccess(Abj, Constant(N));
                for (int ji = j + 1; ji < N; ++ji)
                    r = Add(r, Multiply(ArrayAccess(Abj, Constant(ji)), code[deltas[ji]]));
                code.DeclInit(deltas[j], Divide(Negate(r), ArrayAccess(Abj, _j)));
            }
        }

        // A human readable implementation of RowReduce.
        private static void RowReduce(double[][] Ab, int M, int N)
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
                    double[] Abi = Ab[i];
                    // if(|JxF[i][j]| > max) { pi = i, max = |JxF[i][j]| }
                    double maxj = Math.Abs(Abi[j]);
                    if (maxj > max)
                    {
                        pi = i;
                        max = maxj;
                    }
                }

                // Swap pivot row with the current row.
                if (pi != j)
                {
                    double[] Abpi = Ab[pi];

                    Ab[pi] = Ab[j];
                    Ab[j] = Abpi;
                }

                // Eliminate the rows after the pivot.
                double p = Ab[j][j];
                for (int i = j + 1; i < M; ++i)
                {
                    double[] Abi = Ab[i];
                    double s = Abi[j] / p;
                    if (s != 0.0)
                        for (int ij = j + 1; ij <= N; ++ij)
                            Abi[ij] -= Ab[j][ij] * s;
                }
            }
        }

        // Generate code to perform row reduction.
        private static void RowReduce(CodeGen code, LinqExpr Ab, int M, int N)
        {
            // For each variable in the system...
            for (int j = 0; j + 1 < N; ++j)
            {
                LinqExpr _j = Constant(j);
                LinqExpr Abj = code.ReDeclInit<double[]>("Abj", ArrayAccess(Ab, _j));
                // int pi = j
                LinqExpr pi = code.ReDeclInit<int>("pi", _j);
                // double max = |Ab[j][j]|
                LinqExpr max = code.ReDeclInit<double>("max", Abs(ArrayAccess(Abj, _j)));

                // Find a pivot row for this variable.
                //code.For(j + 1, M, _i =>
                //{
                for (int i = j + 1; i < M; ++i)
                {
                    LinqExpr _i = Constant(i);

                    // if(|Ab[i][j]| > max) { pi = i, max = |Ab[i][j]| }
                    LinqExpr maxj = code.ReDeclInit<double>("maxj", Abs(ArrayAccess(ArrayAccess(Ab, _i), _j)));
                    code.Add(IfThen(
                        GreaterThan(maxj, max),
                        Block(
                            Assign(pi, _i),
                            Assign(max, maxj))));
                }

                // (Maybe) swap the pivot row with the current row.
                LinqExpr Abpi = code.ReDecl<double[]>("Abpi");
                code.Add(IfThen(
                    NotEqual(_j, pi), Block(
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
                LinqExpr p = code.ReDeclInit<double>("p", ArrayAccess(Abj, _j));
                //code.For(j + 1, M, _i =>
                //{
                for (int i = j + 1; i < M; ++i)
                {
                    LinqExpr _i = Constant(i);
                    LinqExpr Abi = code.ReDeclInit<double[]>("Abi", ArrayAccess(Ab, _i));

                    // s = Ab[i][j] / p
                    LinqExpr s = code.ReDeclInit<double>("scale", Divide(ArrayAccess(Abi, _j), p));
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
