using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Reflection;
using ComputerAlgebra;
using ComputerAlgebra.LinqCompiler;
using Circuit;
using Util;
using System.Diagnostics;
using ComputerAlgebra.Plotting;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Jobs;

// Filter design tool: http://sim.okawa-denshi.jp/en/CRtool.php

namespace Tests
{
    class Tester
    {
        private static readonly Variable t = Component.t;

        public int SampleRate = 44100;
        public int Samples = 100000;
        public int Oversample = 4;
        public int Iterations = 8;

        private double _analysisTime;
        private double _simulateTime;

        public Log Log = new ConsoleLog() { Verbosity = MessageType.Info };

        public void Run(IEnumerable<string> Tests, Func<double, double> Vin)
        {
            List<string> errors = new List<string>();
            List<string> performance = new List<string>();

            // This test generates the signal for the LiveSPICE 'logo'.
            //Run("Subcircuit Trivial.schx", Vin, "V1[t]", new Expression[] { "_v15[t]", "_v11[t]" });
            //return;

            foreach (string File in Tests)
            {
                string Name = System.IO.Path.GetFileNameWithoutExtension(File);
                try
                {
                    double perf = Run(File, Vin);
                    performance.Add(Name + ":\t" + Quantity.ToString(perf, Units.Hz) + " (" + (perf / (double)SampleRate).ToString("G3") + "x real time)");
                }
                catch (Exception ex)
                {
                    errors.Add(Name + ":\t" + ex.Message);
                    Log.WriteLine(ex.Message);
                }
            }

            Log.WriteLine("Analyze/Simulate {0}/{1}", _analysisTime, _simulateTime);

            Log.WriteLine("{0} succeeded:", performance.Count);
            foreach (string i in performance)
                Log.WriteLine(i);

            Log.WriteLine("{0} failed:", errors.Count);
            foreach (string i in errors)
                Log.WriteLine(i);
        }

        public double Run(string FileName, Func<double, double> Vin)
        {
            var C = Schematic.Load(FileName, Log).Build();
            C.Name = Path.GetFileNameWithoutExtension(FileName);
            return Run(
                C,
                Vin,
                C.Components.OfType<Input>().Select(i => Expression.Parse(i.Name + "[t]")).DefaultIfEmpty("V[t]").SingleOrDefault(),
                C.Nodes.Select(i => i.V));
        }

        public double Run(string FileName, Func<double, double> Vin, Expression Input, IEnumerable<Expression> Plots)
        {
            Circuit.Circuit C = Schematic.Load(FileName, Log).Build();
            C.Name = System.IO.Path.GetFileNameWithoutExtension(FileName);
            return Run(C, Vin, Input, Plots);
        }

        public double Run(Circuit.Circuit C, Func<double, double> Vin, Expression Input, IEnumerable<Expression> Plots)
        {
            //long a = Timer.Counter;
            var sw = Stopwatch.StartNew();

            Analysis analysis = C.Analyze();
            var transientSolution = TransientSolution.Solve(analysis, (Real)1 / (SampleRate * Oversample), Log);

            _analysisTime += sw.Elapsed.TotalSeconds;

            Simulation S = new Simulation(transientSolution)
            {
                Oversample = Oversample, 
                Iterations = Iterations,
                Log = Log, 
                Inputs = new[] { Input },
                Outputs = Plots.ToArray(),
            };

            Log.WriteLine("");
            return Samples > 0
                ? RunTest(
                    S,
                    Vin,
                    Samples,
                    C.Name)
                : 0.0;
        }

        public double RunTest(Simulation S, Func<double, double> Vin, int Samples, string Name)
        {
            double t0 = (double)S.Time;
            double T = S.TimeStep;

            int N = 353;
            double[] input = new double[N];

            List<List<double>> output = S.Outputs.Select(i => new List<double>(Samples)).ToList();
            List<double[]> buffers = S.Outputs.Select(_ => new double[N]).ToList();
            
            double time = 0.0;
            int samples = 0;
            double t = 0;
            for (; samples < Samples; samples += N)
            {
                for (int n = 0; n < N; ++n, t += T)
                    input[n] = Vin(t);

                var sw = Stopwatch.StartNew();
                S.Run(input, buffers);
                time += sw.Elapsed.TotalSeconds;

                for (int i = 0; i < S.Outputs.Count(); ++i)
                    output[i].AddRange(buffers[i]);
            }
            _simulateTime += time;

            int t1 = Math.Min(samples, 4000);

            Log.WriteLine("Performance {0}", Quantity.ToString(samples / time, Units.Hz));
                        
            Plot p = new Plot()
            {
                Title = Name,
                Width = 800,
                Height = 400,
                x0 = t0,
                x1 = T * t1,
                xLabel = "Time (s)",
                yLabel = "Voltage (V)",
            };

            p.Series.AddRange(output.Select((i, j) => new Scatter(
                i.Take(t1)
                .Select((k, n) => new KeyValuePair<double, double>(n * T, k)).ToArray()) { Name = S.Outputs.ElementAt(j).ToString() }));
            return samples / time;
        }
    }

    //[SimpleJob(RuntimeMoniker.Net472)]
    [SimpleJob(RuntimeMoniker.NetCoreApp31)]
    [MemoryDiagnoser]
    public class Benchmark
    {
        public Log Log = new ConsoleLog() { Verbosity = MessageType.Info };
        private double[] _input;
        private double[][] _buffers;

        //[Params(true, false)]
        public bool Vectorize { get; set; }

        public Simulation S { get; private set; }

        [IterationSetup]
        public void Setup()
        {
            Console.Write(Directory.GetCurrentDirectory());
            var file = Directory.EnumerateFiles("./Circuits", "*.schx").First();
            var C = Schematic.Load(file, Log).Build();

            double Harmonics(double t, double A, double f0, int N)
            {
                double s = 0;
                for (int i = 1; i <= N; ++i)
                    s += Math.Sin(t * f0 * 2 * 3.1415 * i) / N;
                return A * s;
            }

            Func<double,double> inputFun = time => Harmonics(time, 0.5, 82, 2);

            const int sampleRate = 44100;
            const int oversample = 4;
            const int iterations = 8;

            Analysis analysis = C.Analyze();
            var transientSolution = TransientSolution.Solve(analysis, (Real)1 / (sampleRate * oversample), Log);

            S = new Simulation(transientSolution, Vectorize)
            {
                Oversample = oversample,
                Iterations = iterations,
                Log = Log,
                Inputs = new[] { C.Components.OfType<Input>().Select(i => Expression.Parse(i.Name + "[t]")).DefaultIfEmpty("V[t]").SingleOrDefault() },
                Outputs = C.Nodes.Select(i => i.V).ToArray()
            };


            _input = Enumerable.Range(0, 10000).Select(i => inputFun(i * S.TimeStep)).ToArray();
            _buffers = S.Outputs.Select(_ => new double[10000]).ToArray();

        }

        [Benchmark(Baseline = true)]
        public void TestPerformance()
        {
            S.Run(_input, _buffers);
        }
    }

    [SimpleJob(RuntimeMoniker.Net472)]
    [SimpleJob(RuntimeMoniker.NetCoreApp31)]
    public class Powers
    {
        [Benchmark(Baseline = true)]
        public double StandardPower() => Math.Pow(1.1d, 2.2d);

        [Benchmark]
        public double StandardMathPower() => StandardMath.Pow(1.1d, 2.2d);

        [Benchmark]
        public double LogPower() => Math.Exp(Math.Log(1.1d) * 2.2d);
    }

    class Program
    {
        static void WriteDocs()
        {
            FileStream file = new FileStream("docs.html", FileMode.Create);
            StreamWriter docs = new StreamWriter(file);

            Action<StreamWriter, string, string, string> WriteTag = (StreamWriter S, string Tab, string Tag, string P) => S.WriteLine(Tab + "<" + Tag + ">" + P + "</" + Tag + ">");

            docs.WriteLine("<section id=\"components\">");
            docs.WriteLine("\t<h3>Components</h3>");
            docs.WriteLine("\t<p>This section describes the operation of the basic component types provided in LiveSPICE. All of the components in the library are directly or indirectly (via subcircuits) implemented using these component types.</p>");

            Type root = typeof(Circuit.Component);
            foreach (Assembly i in AppDomain.CurrentDomain.GetAssemblies())
            {
                foreach (Type j in i.GetTypes().Where(j => j.IsPublic && !j.IsAbstract && root.IsAssignableFrom(j)))
                {
                    try
                    {
                        System.ComponentModel.DisplayNameAttribute name = j.CustomAttribute<System.ComponentModel.DisplayNameAttribute>();
                        System.ComponentModel.DescriptionAttribute desc = j.CustomAttribute<System.ComponentModel.DescriptionAttribute>();
                        
                        docs.WriteLine("\t<section id=\"" + j.Name + "\">");
                        docs.WriteLine("\t<h4>" + (name != null ? name.DisplayName : j.Name) + "</h4>");
                        if (desc != null)
                            WriteTag(docs, "\t\t", "p", desc.Description);

                        docs.WriteLine("\t\t<h5>Properties</h5>");
                        docs.WriteLine("\t\t<ul>");
                        foreach (PropertyInfo p in j.GetProperties(BindingFlags.Public | BindingFlags.Instance).Where(k => k.CustomAttribute<Serialize>() != null))
                        {
                            desc = p.CustomAttribute<System.ComponentModel.DescriptionAttribute>();
                            StringBuilder prop = new StringBuilder();
                            prop.Append("<span class=\"property\">" + p.Name + "</span>");
                            if (desc != null)
                                prop.Append(": " + desc.Description);

                            WriteTag(docs, "\t\t\t", "li", prop.ToString());
                        }
                        docs.WriteLine("\t\t</ul>");
                    }
                    catch (Exception) { }
                }
            }

            docs.WriteLine("</section> <!-- components -->");
        }

        static void Main(string[] args)
        {
            BenchmarkRunner.Run<Powers>();
            //WriteDocs();

            //Tester tests = new Tester();
            //tests.Run(System.IO.Directory.EnumerateFiles(@".", "*.schx"), t => Harmonics(t, 0.5, 82, 2));
        }

        // Generate a function with the first N harmonics of f0.
        static double Harmonics(double t, double A, double f0, int N)
        {
            double s = 0;
            for (int i = 1; i <= N; ++i)
                s += Math.Sin(t * f0 * 2 * 3.1415 * i) / N;
            return A * s;
        }
    }
}
