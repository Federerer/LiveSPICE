﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Reflection.Emit;
using SyMath;
using LinqExpressions = System.Linq.Expressions;
using LinqExpression = System.Linq.Expressions.Expression;

namespace Circuit
{
    /// <summary>
    /// Simulate a circuit.
    /// </summary>
    public class Simulation
    {
        // Holds a T instance and a LINQ expression that maps to the instance.
        class GlobalExpr<T>
        {
            private T x;
            public T Value { get { return x; } set { x = value; } }

            // A Linq Expression to refer to the voltage at this node.
            private LinqExpression expr;
            public LinqExpression Expr { get { return expr; } }

            public GlobalExpr() { expr = LinqExpression.Field(LinqExpression.Constant(this), typeof(GlobalExpr<T>), "x"); }
            public GlobalExpr(T Init) : this() { x = Init; }
        }

        // Stores the stateful data associated with a node in the circuit simulation.
        class Node
        {
            // Expression describing a single timestep.
            public Expression Step;

            // Current voltage at this node.
            private GlobalExpr<double> v = new GlobalExpr<double>();
            public double V { get { return v.Value; } set { v.Value = value; } }
            public LinqExpression VExpr { get { return v.Expr; } }

            public Node(Expression Step) { this.Step = Step; }
        }

        // Nodes in this simulation.
        private Dictionary<Expression, Node> nodes;
                
        // Store the previous inputs for interpolation.
        private Dictionary<Expression, GlobalExpr<double>> inputs = new Dictionary<Expression, GlobalExpr<double>>();

        // Current time of the simulation (Component.t).
        protected Expression _t = Constant.Zero;
        public Expression t { get { return _t; } }

        // Timestep of the simulation (Component.T).
        protected Expression _T;
        public Expression T { get { return _T; } }

        // Expression for t at the previous timestep.
        private static Expression t0 = Variable.New("t0");

        /// <summary>
        /// Get the nodes being simulated.
        /// </summary>
        public IEnumerable<Expression> Nodes { get { return nodes.Keys; } }

        protected int iterations = 1;
        /// <summary>
        /// Get or set the maximum number of iterations to use when numerically solving equations.
        /// </summary>
        public int Iterations { get { return iterations; } set { iterations = value; } }
        protected int oversample = 1;
        /// <summary>
        /// Get or set the oversampling factor for the simulation.
        /// </summary>
        public int Oversample { get { return oversample; } set { oversample = value; } }

        // Shorthand for df/dx.
        private static Expression D(Expression f, Expression x) { return f.Differentiate(x); }
        // Solve a system of equations, possibly including linear differential equations.
        private static List<Arrow> Solve(List<Equal> f, List<Expression> y, Expression t, Expression t0, Expression h, IntegrationMethod method, int iterations)
        {
            List<Expression> dydt = y.Select(i => D(i, t)).ToList();

            List<Arrow> step = new List<Arrow>();
            // Try solving for y algebraically.
            List<Arrow> linear = f.Solve(y);
            // Only accept independent solutions.
            linear.RemoveAll(i => i.Right.IsFunctionOf(dydt.Append(i.Left)));
            step.AddRange(linear);
            // Substitute the solutions found.
            f = f.Evaluate(linear).OfType<Equal>().ToList();
            y.RemoveAll(i => step.Find(j => j.Left.Equals(i)) != null);
            
            // Solve for any non-differential functions remaining and substitute them into the system.
            List<Arrow> nondifferential = f.Where(i => !i.IsFunctionOf(dydt)).Solve(y);
            List<Equal> df = f.Evaluate(nondifferential).OfType<Equal>().ToList();
            // Solve the differential equations.
            List<Arrow> differentials = df.NDSolve(y, t, t0, h, method, iterations);
            step.AddRange(differentials);
            // Replace differentials with approximations.
            f = f.Evaluate(y.Select(i => Arrow.New(D(i, t), (i - i.Evaluate(t, t0)) / h))).Cast<Equal>().ToList();
            f = f.Evaluate(differentials).OfType<Equal>().ToList();
            y.RemoveAll(i => step.Find(j => j.Left.Equals(i)) != null);

            // Try solving for numerical solutions.
            List<Arrow> nonlinear = f.NSolve(y.Select(i => Arrow.New(i, i.Evaluate(t, t0))), iterations);
            nonlinear.RemoveAll(i => i.Right.IsFunctionOf(dydt));
            step.AddRange(nonlinear);

            return step;
        }

        /// <summary>
        /// Create a simulation for the given circuit.
        /// </summary>
        /// <param name="C">Circuit to simulate.</param>
        /// <param name="T">Sampling period.</param>
        /// <returns></returns>
        public Simulation(Circuit C, Quantity F)
        {
            _T = 1.0 / ((Expression)F * Oversample);

            // Compute the KCL equations for this circuit.
            List<Equal> kcl = C.Analyze();
            
            // Find the expression for the next timestep via the trapezoid method (ala bilinear transform).
            List<Arrow> step = Solve(
                kcl.ToList(),
                C.Nodes.Select(i => i.V).Cast<Expression>().ToList(), 
                Component.t, t0, 
                _T, 
                IntegrationMethod.Trapezoid, 
                iterations);

            // Create the nodes.
            nodes = step.ToDictionary(
                i => (Expression)Call.New(((Call)i.Left).Target, Component.t), 
                i => new Node(i.Right));
        }

        public void Reset()
        {
            _t = Constant.Zero;
            foreach (Node i in nodes.Values)
                i.V = 0.0;
            foreach (GlobalExpr<double> i in inputs.Values)
                i.Value = 0.0;
        }
        
        // Process some samples. Requested nodes are stored in Output.
        public void Process(int N, IDictionary<Expression, double[]> Input, IDictionary<Expression, double[]> Output, IEnumerable<Arrow> Arguments)
        {
            Delegate processor = Compile(Input.Keys, Output.Keys, Arguments.Select(i => i.Left));

            // Build parameter list for the processor.
            List<object> parameters = new List<object>();
            parameters.Add(N);
            parameters.Add((double)t);
            parameters.Add((double)T);
            foreach (KeyValuePair<Expression, double[]> i in Input)
                parameters.Add(i.Value);
            foreach (KeyValuePair<Expression, double[]> i in Output)
                parameters.Add(i.Value);
            foreach (Arrow i in Arguments)
                parameters.Add((double)i.Right);

            _t = (double)processor.DynamicInvoke(parameters.ToArray());
        }

        public void Process(int N, IDictionary<Expression, double[]> Input, IDictionary<Expression, double[]> Output, params Arrow[] Arguments)
        {
            Process(N, Input, Output, Arguments.AsEnumerable());
        }

        public void Process(Expression InputNode, double[] InputSamples, IDictionary<Expression, double[]> Output)
        {
            Process(
                InputSamples.Length,
                new Dictionary<Expression, double[]>() { { InputNode, InputSamples } },
                Output);
        }

        public void Process(Expression InputNode, double[] InputSamples, Expression OutputNode, double[] OutputSamples)
        {
            Process(
                InputSamples.Length,
                new Dictionary<Expression, double[]>() { { InputNode, InputSamples } },
                new Dictionary<Expression, double[]>() { { OutputNode, OutputSamples } });
        }

        // Compile and cache delegates for processing various IO configurations for this simulation.
        Dictionary<long, Delegate> compiled = new Dictionary<long, Delegate>();
        private Delegate Compile(IEnumerable<Expression> Input, IEnumerable<Expression> Output, IEnumerable<Expression> Parameters)
        {
            long hash = Input.OrderedHashCode();
            hash = hash * 33 + Output.OrderedHashCode();
            hash = hash * 33 + Parameters.OrderedHashCode();

            Delegate d;
            if (compiled.TryGetValue(hash, out d))
                return d;

            d = ProcessExpression(Input, Output, Parameters).Compile();
            compiled[hash] = d;
            return d;
        }
        
        // The resulting lambda processes N samples, using buffers provided for Input and Output.
        // Arguments: int N, double t, double T, double[] x Input, double[] x Output, double[] x Arguments
        // Returns: t + N * T
        private LinqExpressions.LambdaExpression ProcessExpression(IEnumerable<Expression> Input, IEnumerable<Expression> Output, IEnumerable<Expression> Parameters)
        {
            // Map expressions to identifiers in the syntax tree.
            Dictionary<Expression, LinqExpression> v = new Dictionary<Expression, LinqExpression>();
            Dictionary<Expression, LinqExpression> buffers = new Dictionary<Expression, LinqExpression>();

            // Get expressions for the state of each node. These may be replaced by input parameters.
            foreach (KeyValuePair<Expression, Node> i in nodes)
                v[i.Key.Evaluate(Component.t, t0)] = i.Value.VExpr;

            // Lambda definition objects.
            LinqExpressions.LabelTarget returnTo = LinqExpression.Label(typeof(double));
            List<LinqExpressions.ParameterExpression> parameters = new List<LinqExpressions.ParameterExpression>();
            List<LinqExpression> body = new List<LinqExpression>();
            List<LinqExpressions.ParameterExpression> locals = new List<LinqExpressions.ParameterExpression>();

            // Parameters to the lambda.
            LinqExpressions.ParameterExpression pN = LinqExpression.Parameter(typeof(int), "N");
            parameters.Add(pN);

            LinqExpressions.ParameterExpression pt0 = LinqExpression.Parameter(typeof(double), "t0");
            parameters.Add(pt0);
            v[t0] = pt0;

            LinqExpressions.ParameterExpression pT = LinqExpression.Parameter(typeof(double), "T");
            parameters.Add(pT);
            v[Component.T] = pT;

            foreach (Expression i in Input)
            {
                LinqExpressions.ParameterExpression arg = LinqExpression.Parameter(typeof(double[]), i.ToString());
                parameters.Add(arg);
                buffers[i] = arg;
            }

            foreach (Expression i in Output)
            {
                LinqExpressions.ParameterExpression arg = LinqExpression.Parameter(typeof(double[]), i.ToString());
                parameters.Add(arg);
                buffers[i] = arg;
            }

            foreach (Expression i in Parameters)
            {
                LinqExpressions.ParameterExpression arg = LinqExpression.Parameter(typeof(double), i.ToString());
                parameters.Add(arg);
                v[i] = arg;
            }
            
            // Defining lambda body.

            // Set t = t0.
            LinqExpressions.ParameterExpression vt = LinqExpression.Variable(typeof(double), "t");
            body.Add(LinqExpression.Assign(vt, pt0));
            locals.Add(vt);
            v[Component.t] = vt;
            
            // Set n = 0.
            LinqExpressions.ParameterExpression vn = LinqExpression.Variable(typeof(int), "n");
            locals.Add(vn);
            body.Add(LinqExpression.Assign(vn, LinqExpression.Constant(0)));

            // N for loop header.
            LinqExpressions.LabelTarget forN = LinqExpression.Label("forN");
            body.Add(LinqExpression.Label(forN));
            body.Add(LinqExpression.IfThen(
                LinqExpression.GreaterThanOrEqual(vn, pN),
                LinqExpression.Return(returnTo, vt, typeof(double))));

            // N for loop body.
            Dictionary<Expression, LinqExpression> dinput = new Dictionary<Expression, LinqExpression>();
            // Get input samples.
            foreach (Expression i in Input)
            {
                // Ensure that we have a variable to store the previous sample in.
                inputs[i] = new GlobalExpr<double>(0.0);
                LinqExpression va = inputs[i].Expr;
                LinqExpression vb = LinqExpression.MakeIndex(
                    buffers[i], 
                    typeof(double[]).GetProperty("Item"), 
                    new LinqExpression[] { vn });

                // Create a local variable for the interpolated sample and assign va to it.
                LinqExpressions.ParameterExpression vi = LinqExpression.Variable(typeof(double), i.ToString() + "_i");
                locals.Add(vi);
                v[i] = vi;
                body.Add(LinqExpression.Assign(vi, va));
                
                // Compute the delta for the sample per oversample iteration.
                LinqExpressions.ParameterExpression dinputi = LinqExpression.Variable(typeof(double), "d" + i.ToString());
                locals.Add(dinputi);
                dinput[i] = dinputi;
                body.Add(LinqExpression.Assign(dinputi, LinqExpression.Divide(LinqExpression.Subtract(vb, va), LinqExpression.Constant((double)Oversample))));

                // The next sample's va = this samples vb.
                body.Add(LinqExpression.Assign(va, vb));

                // If i isn't a node, just make a dummy expression for the previous timestep. 
                // This might be able to be removed with an improved system solver that doesn't create references to i[t0] when i is not a node.
                if (!nodes.ContainsKey(i))
                    v[i.Evaluate(Component.t, t0)] = v[i];
            }

            // Oversampling loop header.
            LinqExpressions.ParameterExpression ov = LinqExpression.Variable(typeof(int), "ov");
            locals.Add(ov);
            body.Add(LinqExpression.Assign(ov, LinqExpression.Constant(Oversample)));
            LinqExpressions.LabelTarget forOv = LinqExpression.Label("forOv");
            body.Add(LinqExpression.Label(forOv));

            // Set t = t0 + T.
            body.Add(LinqExpression.Assign(vt, LinqExpression.Add(pt0, pT)));
            
            // Interpolate the input functions.
            foreach (Expression i in Input)
                body.Add(LinqExpression.AddAssign(v[i], dinput[i]));

            // Compile step expressions and assign to the node state.
            foreach (KeyValuePair<Expression, Node> i in nodes)
                body.Add(LinqExpression.Assign(i.Value.VExpr, i.Value.Step.Compile(v)));

            // Update t0 = t.
            body.Add(LinqExpression.Assign(pt0, vt));

            // If --ov > 0, go back to oversampling loop header.
            body.Add(LinqExpression.IfThen(
                LinqExpression.GreaterThan(LinqExpression.PreDecrementAssign(ov), LinqExpression.Constant(0)),
                LinqExpression.Goto(forOv)));
            
            // Store output samples.
            foreach (Expression i in Output)
            {
                Node n;
                nodes.TryGetValue(i, out n);
                body.Add(LinqExpression.Assign(
                    LinqExpression.MakeIndex(
                        buffers[i],
                        typeof(double[]).GetProperty("Item"),
                        new LinqExpression[] { vn }),
                    n != null ? n.VExpr : LinqExpression.Constant(double.NaN)));
            }
            
            // ++n.
            body.Add(LinqExpression.PreIncrementAssign(vn));

            // Go to the beginning of the loop over N.
            body.Add(LinqExpression.Goto(forN));
            
            body.Add(LinqExpression.Label(returnTo, vt));
            return LinqExpression.Lambda(LinqExpression.Block(locals, body), parameters);
        }
    }
}
