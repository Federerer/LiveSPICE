using System;
using System.ComponentModel;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using ComputerAlgebra;
using ComputerAlgebra.LinqCompiler;
using Util;
using System.Diagnostics;
using System.IO;
using Audio;
using System.Numerics;
using Vector = System.Numerics.Vector;
using SignalProcessing;
using AvalonDock.Layout;

namespace LiveSPICE
{
    /// <summary>
    /// Interaction logic for LiveSimulation.xaml
    /// </summary>
    partial class LiveSimulation : Window, INotifyPropertyChanged
    {
        public Log Log { get { return (Log)log.Content; } }
        public Scope Scope { get { return (Scope)scope.Content; } }

        protected int oversample = 2;
        /// <summary>
        /// Simulation oversampling rate.
        /// </summary>
        public int Oversample
        {
            get { return oversample; }
            set { oversample = value; RebuildSolution(); NotifyChanged("Oversample"); }
        }

        protected int iterations = 8;
        /// <summary>
        /// Max iterations for numerical algorithms.
        /// </summary>
        public int Iterations
        {
            get { return iterations; }
            set { iterations = value; RebuildSolution(); NotifyChanged("Iterations"); }
        }

        private double inputGain = 1.0;
        /// <summary>
        /// Overall input gain.
        /// </summary>
        public double InputGain { get { return inputGain; } set { inputGain = value; NotifyChanged("InputGain"); } }

        private double outputGain = 1.0;
        /// <summary>
        /// Overall output gain.
        /// </summary>
        public double OutputGain { get { return outputGain; } set { outputGain = value; NotifyChanged("OutputGain"); } }

        private SimulationSchematic Schematic { get { return (SimulationSchematic)schematic.Schematic; } set { schematic.Schematic = value; } }

        protected Circuit.Circuit circuit;
        protected Circuit.Simulation simulation;

        private List<Probe> probes = new List<Probe>();

        private object sync = new object();

        protected Audio.Stream stream = null;

        protected ObservableCollection<InputChannel> _inputChannels = new ObservableCollection<InputChannel>();
        protected ObservableCollection<OutputChannel> _outputChannels = new ObservableCollection<OutputChannel>();
        public ObservableCollection<InputChannel> InputChannels { get { return _inputChannels; } }
        public ObservableCollection<OutputChannel> OutputChannels { get { return _outputChannels; } }

        private Dictionary<ComputerAlgebra.Expression, Channel> inputs = new Dictionary<ComputerAlgebra.Expression, Channel>();

        public string SelectedImpulse { get; set; } = "orange_2_mics.wav";

        public bool Bypass { get; set; }

        public LiveSimulation(Circuit.Schematic Simulate, Device Device, Audio.Channel[] Inputs, Audio.Channel[] Outputs)
        {
            try
            {
                InitializeComponent();

                this.DataContext = this;

                // Make a clone of the schematic so we can mess with it.
                Circuit.Schematic clone = Circuit.Schematic.Deserialize(Simulate.Serialize(), Log);
                clone.Elements.ItemAdded += OnElementAdded;
                clone.Elements.ItemRemoved += OnElementRemoved;
                Schematic = new SimulationSchematic(clone);
                Schematic.SelectionChanged += OnProbeSelected;

                // Build the circuit from the schematic.
                circuit = Schematic.Schematic.Build(Log);

                // Create the input and output controls.                
                IEnumerable<Circuit.Component> components = circuit.Components;

                // Create audio input channels.
                for (int i = 0; i < Inputs.Length; ++i)
                    InputChannels.Add(new InputChannel(i) { Name = Inputs[i].Name });

                ComputerAlgebra.Expression speakers = 0;

                foreach (var component in components)
                {
                    if (!(component.Tag is Circuit.Symbol S))
                        continue;

                    SymbolControl tag = (SymbolControl)S.Tag;
                    if (tag == null)
                        continue;

                    // Create potentiometers.
                    switch (component)
                    {
                        case Circuit.IPotControl c:
                            {
                                PotControl pot = new PotControl()
                                {
                                    Width = 80,
                                    Height = 80,
                                    Opacity = 0.5,
                                    FontSize = 15,
                                    FontWeight = FontWeights.Bold,
                                };
                                Schematic.overlays.Children.Add(pot);
                                Canvas.SetLeft(pot, Canvas.GetLeft(tag) - pot.Width / 2 + tag.Width / 2);
                                Canvas.SetTop(pot, Canvas.GetTop(tag) - pot.Height / 2 + tag.Height / 2);

                                pot.Value = c.PotValue;
                                pot.ValueChanged += x => { c.PotValue = x; UpdateSimulation(false); };

                                pot.MouseEnter += (o, e) => pot.Opacity = 0.95;
                                pot.MouseLeave += (o, e) => pot.Opacity = 0.5;
                                break;
                            }
                        case Circuit.IButtonControl b:
                            {
                                Button button = new Button()
                                {
                                    Width = tag.Width,
                                    Height = tag.Height,
                                    Opacity = 0.5,
                                    Background = Brushes.White,
                                };
                                Schematic.overlays.Children.Add(button);
                                Canvas.SetLeft(button, Canvas.GetLeft(tag));
                                Canvas.SetTop(button, Canvas.GetTop(tag));

                                button.Click += (o, e) =>
                                {
                                    // Click all the buttons in the group.
                                    foreach (Circuit.IButtonControl j in components.OfType<Circuit.IButtonControl>().Where(x => x.Group == b.Group))
                                        j.Click();
                                    UpdateSimulation(true);
                                };

                                button.MouseEnter += (o, e) => button.Opacity = 0.95;
                                button.MouseLeave += (o, e) => button.Opacity = 0.5;
                                break;
                            }
                        case Circuit.Speaker output:
                            {
                                speakers += output.V;
                                break;
                            }
                        case Circuit.Input input:
                            {
                                tag.ShowText = false;

                                ComboBox combo = new ComboBox()
                                {
                                    Width = 80,
                                    Height = 24,
                                    Opacity = 0.5,
                                    IsEditable = true,
                                    SelectedValuePath = "Tag",
                                };

                                foreach (InputChannel j in InputChannels)
                                {
                                    combo.Items.Add(new ComboBoxItem()
                                    {
                                        Tag = j,
                                        Content = j.Name
                                    });
                                }

                                Schematic.overlays.Children.Add(combo);
                                Canvas.SetLeft(combo, Canvas.GetLeft(tag) - combo.Width / 2 + tag.Width / 2);
                                Canvas.SetTop(combo, Canvas.GetTop(tag) - combo.Height / 2 + tag.Height / 2);

                                ComputerAlgebra.Expression V = Circuit.Component.DependentVariable(input.Name, Circuit.Component.t);
                                inputs[V] = new SignalChannel(0);

                                combo.SelectionChanged += (o, e) =>
                                {
                                    if (combo.SelectedItem != null)
                                    {
                                        ComboBoxItem it = (ComboBoxItem)combo.SelectedItem;
                                        inputs[V] = new InputChannel(((InputChannel)it.Tag).Index);
                                    }
                                };

                                combo.AddHandler(TextBox.KeyDownEvent, new KeyEventHandler((o, e) =>
                                {
                                    try
                                    {
                                        inputs[V] = new SignalChannel(Circuit.Quantity.Parse(combo.Text, Circuit.Units.V));
                                    }
                                    catch (Exception)
                                    {
                                        // If there is an error in the expression, zero out the signal.
                                        inputs[V] = new SignalChannel(0);
                                    }
                                }));

                                if (combo.Items.Count > 0)
                                    combo.SelectedItem = combo.Items[0];
                                else
                                    combo.Text = "0 V";

                                combo.MouseEnter += (o, e) => combo.Opacity = 0.95;
                                combo.MouseLeave += (o, e) => combo.Opacity = 0.5;
                                break;
                            }
                    }
                }

                // Create audio output channels.
                for (int i = 0; i < Outputs.Length; ++i)
                {
                    OutputChannel c = new OutputChannel(i) { Name = Outputs[i].Name, Signal = speakers };
                    c.PropertyChanged += (o, e) => { if (e.PropertyName == "Signal") RebuildSolution(); };
                    OutputChannels.Add(c);
                }

                // Begin audio processing.
                if (Inputs.Length > 0 || Outputs.Length > 0)
                    stream = Device.Open(ProcessSamples, Inputs, Outputs);
                else
                    stream = new NullStream(ProcessSamples);

                ContentRendered += (o, e) => RebuildSolution();

                Closed += (s, e) => stream.Stop();
            }
            catch (Exception Ex)
            {
                MessageBox.Show(Ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void RebuildSolution()
        {
            lock (sync)
            {
                simulation = null;
                ProgressDialog.RunAsync(this, "Building circuit solution...", () =>
                {
                    try
                    {
                        var h = (ComputerAlgebra.Expression)1 / (stream.SampleRate * Oversample);
                        var solution = Circuit.TransientSolution.Solve(circuit.Analyze(), h, Log);

                        simulation = new Circuit.Simulation(solution, inputs.Keys.ToArray(), probes.Select(i => i.V).Concat(OutputChannels.Select(i => i.Signal)).ToArray(), Oversample, Iterations);
                    }
                    catch (Exception Ex)
                    {
                        Log.WriteException(Ex);
                    }
                });
            }
        }

        private int clock = -1;
        private int update = 0;
        private TaskScheduler scheduler = new RedundantTaskScheduler(1);
        private void UpdateSimulation(bool rebuild)
        {
            int id = Interlocked.Increment(ref update);
            new Task(() =>
            {
                ComputerAlgebra.Expression h = (ComputerAlgebra.Expression)1 / (stream.SampleRate * Oversample);
                Circuit.TransientSolution s = Circuit.TransientSolution.Solve(circuit.Analyze(), h, rebuild ? (ILog)Log : new NullLog());
                lock (sync)
                {
                    if (id > clock)
                    {
                        if (rebuild)
                        {
                            simulation = new Circuit.Simulation(s, inputs.Keys.ToArray(), probes.Select(i => i.V).Concat(OutputChannels.Select(i => i.Signal)).ToArray(), Oversample, Iterations);
                        }
                        else
                        {
                            simulation.Solution = s;
                            clock = id;
                        }
                    }
                }
            }).Start(scheduler);
        }

        private void ProcessSamples(int Count, SampleBuffer[] In, SampleBuffer[] Out, double Rate)
        {
            // Apply input gain.
            for (int i = 0; i < In.Length; ++i)
            {
                Channel ch = InputChannels[i];
                double peak = In[i].Amplify(inputGain * ch.V0dB);
                Dispatcher.InvokeAsync(() => ch.SignalStatus = MapSignalToBrush(peak));
            }

            // Run the simulation.
            lock (sync)
            {
                if (simulation != null)
                    RunSimulation(Count, In, Out, Rate);
                else
                    foreach (SampleBuffer i in Out)
                        i.Clear();
            }

            // Apply output gain.
            for (int i = 0; i < Out.Length; ++i)
            {
                Channel ch = OutputChannels[i];
                double peak = Out[i].Amplify(outputGain / ch.V0dB);
                Dispatcher.InvokeAsync(() => ch.SignalStatus = MapSignalToBrush(peak));
            }

            // Tick oscilloscope.
            Scope.Signals.TickClock(Count, Rate);
        }

        private void RunSimulation(int Count, SampleBuffer[] In, SampleBuffer[] Out, double Rate)
        {
            try
            {
                // If the sample rate changed, we need to kill the simulation and let the foreground rebuild it.
                if (Rate != (double)simulation.SampleRate)
                {
                    simulation = null;
                    Dispatcher.InvokeAsync(RebuildSolution);
                    return;
                }

                List<double[]> ins = new List<double[]>(inputs.Count);
                foreach (Channel i in inputs.Values)
                {
                    if (i is InputChannel)
                        ins.Add(In[((InputChannel)i).Index].LockSamples(true, false));
                    else if (i is SignalChannel)
                        ins.Add(((SignalChannel)i).Buffer(Count, simulation.Time, simulation.TimeStep));
                }

                var hwOuts = Out.Select(o => o.LockSamples(false, true)).ToArray();
                var probeOuts = probes.Select(p => p.AllocBuffer(Count)).ToArray();

                // Process the samples!
                simulation.Run(Count, ins, probeOuts.Concat(hwOuts));

                foreach (var outputChannel in OutputChannels)
                {
                    if (outputChannel.FilterEnabled)
                    {
                        outputChannel.Filter.ProcessSamples(hwOuts[outputChannel.Index]);
                    }
                }

                // Show the samples on the oscilloscope.
                long clock = Scope.Signals.Clock;
                foreach (Probe i in probes)
                    i.Signal.AddSamples(clock, i.Buffer);
            }
            catch (Circuit.SimulationDiverged Ex)
            {
                // If the simulation diverged more than one second ago, reset it and hope it doesn't happen again.
                Log.WriteLine(MessageType.Error, "Error: " + Ex.Message);
                simulation = null;
                if (Ex.At > Rate)
                    Dispatcher.InvokeAsync(RebuildSolution);
                foreach (var i in Out)
                    i.Clear();
            }
            catch (Exception Ex)
            {
                // If there was a more serious error, kill the simulation so the user can fix it.
                Log.WriteException(Ex);
                simulation = null;
                foreach (SampleBuffer i in Out)
                    i.Clear();
            }

            // Unlock sample buffers.
            foreach (SampleBuffer i in Out)
                i.Unlock();
            foreach (SampleBuffer i in In)
                i.Unlock();
        }

        private static double AmplifySignal(SampleBuffer Signal, double Gain)
        {
            double peak = 0.0;
            using (SamplesLock samples = new SamplesLock(Signal, true, true))
            {
                for (int i = 0; i < samples.Count; ++i)
                {
                    double v = samples[i];
                    v *= Gain;
                    peak = Math.Max(peak, Math.Abs(v));
                    samples[i] = v;
                }
            }
            return peak;
        }

        private void OnElementAdded(object sender, Circuit.ElementEventArgs e)
        {
            if (e.Element is Circuit.Symbol && ((Circuit.Symbol)e.Element).Component is Probe)
            {
                Probe probe = (Probe)((Circuit.Symbol)e.Element).Component;
                probe.Signal = new Signal()
                {
                    Name = probe.V.ToString(),
                    Pen = MapToSignalPen(probe.Color)
                };
                Scope.Signals.Add(probe.Signal);
                Scope.SelectedSignal = probe.Signal;
                lock (sync)
                {
                    probes.Add(probe);
                    if (simulation != null)
                        simulation.Outputs = probes.Select(i => i.V).Concat(OutputChannels.Select(i => i.Signal)).ToArray();
                }
            }
        }

        private void OnElementRemoved(object sender, Circuit.ElementEventArgs e)
        {
            if (e.Element is Circuit.Symbol && ((Circuit.Symbol)e.Element).Component is Probe)
            {
                Probe probe = (Probe)((Circuit.Symbol)e.Element).Component;
                Scope.Signals.Remove(probe.Signal);
                lock (sync)
                {
                    probes.Remove(probe);
                    if (simulation != null)
                        simulation.Outputs = probes.Select(i => i.V).Concat(OutputChannels.Select(i => i.Signal)).ToArray();
                }
            }
        }

        private void OnProbeSelected(object sender, EventArgs e)
        {
            IEnumerable<Circuit.Symbol> selected = SimulationSchematic.ProbesOf(Schematic.Selected);
            if (selected.Any())
                Scope.SelectedSignal = ((Probe)selected.First().Component).Signal;
        }

        private void Simulate_Executed(object sender, ExecutedRoutedEventArgs e) { RebuildSolution(); }

        private void Exit_Executed(object sender, ExecutedRoutedEventArgs e) { Close(); }

        private void ViewScope_Click(object sender, RoutedEventArgs e) { ToggleVisible(scope); }
        private void ViewAudio_Click(object sender, RoutedEventArgs e) { ToggleVisible(audio); }
        private void ViewLog_Click(object sender, RoutedEventArgs e) { ToggleVisible(log); }

        private void BindSignal_Click(object sender, RoutedEventArgs e)
        {
            OutputChannel ch = (OutputChannel)((FrameworkElement)sender).Tag;

            SchematicTool tool = Schematic.Tool;

            Schematic.Tool = new FindRelevantTool(Schematic)
            {
                Relevant = (x) => x is Circuit.Symbol && ((Circuit.Symbol)x).Component is Circuit.TwoTerminal,
                Clicked = (x) =>
                {
                    if (x.Any())
                    {
                        ComputerAlgebra.Expression init = (Keyboard.Modifiers & ModifierKeys.Control) != 0 ? ch.Signal : 0;
                        ch.Signal = x.OfType<Circuit.Symbol>()
                            .Select(i => i.Component)
                            .OfType<Circuit.TwoTerminal>()
                            .Aggregate(init, (sum, c) => sum + c.V);
                    }
                    Schematic.Tool = tool;
                }
            };
        }

        private static void ToggleVisible(LayoutAnchorable Anchorable)
        {
            if (Anchorable.IsVisible)
                Anchorable.Hide();
            else
                Anchorable.Show();
        }

        private static Pen MapToSignalPen(Circuit.EdgeType Color)
        {
            switch (Color)
            {
                // These two need to be brighter than the normal colors.
                case Circuit.EdgeType.Red: return new Pen(new SolidColorBrush(System.Windows.Media.Color.FromRgb(255, 80, 80)), 1.0);
                case Circuit.EdgeType.Blue: return new Pen(new SolidColorBrush(System.Windows.Media.Color.FromRgb(20, 180, 255)), 1.0);
                default: return ElementControl.MapToPen(Color);
            }
        }

        private static Brush MapSignalToBrush(double Peak)
        {
            if (Peak < 0.5) return Brushes.Green;
            if (Peak < 0.75) return Brushes.Yellow;
            if (Peak < 0.95) return Brushes.Orange;
            return Brushes.Red;
        }

        // INotifyPropertyChanged.
        private void NotifyChanged(string p)
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(p));
        }
        public event PropertyChangedEventHandler PropertyChanged;
    }

    class WavFile
    {
        public int samplesPerSecond { get; set; }
        public int samplesTotalCount { get; set; }
        public string wavFilename { get; private set; }
        public byte[] metaData { get; set; }


        public WavFile(string filename)
        {
            samplesTotalCount = samplesPerSecond = -1;
            wavFilename = filename;
            metaData = null;
        }


        /// <summary>
        /// Reading all samples of a 16-bit stereo wav file into arrays.
        /// </summary>

        public bool readData(ref double[] L)
        {
            try
            {
                BinaryReader reader = new BinaryReader(File.Open(wavFilename, FileMode.Open));

                // header (8 + 4 bytes):

                byte[] riffId = reader.ReadBytes(4);    // "RIFF"
                int fileSize = reader.ReadInt32();      // size of entire file
                byte[] typeId = reader.ReadBytes(4);    // "WAVE"

                if (Encoding.ASCII.GetString(typeId) != "WAVE") return false;

                // chunk 1 (8 + 16 or 18 bytes):

                byte[] fmtId = reader.ReadBytes(4);     // "fmt "
                int fmtSize = reader.ReadInt32();       // size of chunk in bytes
                int fmtCode = reader.ReadInt16();       // 1 - for PCM
                int channels = reader.ReadInt16();      // 1 - mono, 2 - stereo
                int sampleRate = reader.ReadInt32();    // sample rate per second
                int byteRate = reader.ReadInt32();      // bytes per second
                int dataAlign = reader.ReadInt16();     // data align
                int bitDepth = reader.ReadInt16();      // 8, 16, 24, 32, 64 bits

                if (fmtCode != 1) return false;     // not PCM
                if (channels != 1) return false;    // only Mono files in this version
                if (bitDepth != 24) return false;   // only 16-bit in this version

                if (fmtSize == 18) // fmt chunk can be 16 or 18 bytes
                {
                    int fmtExtraSize = reader.ReadInt16();  // read extra bytes size
                    reader.ReadBytes(fmtExtraSize);         // skip over "INFO" chunk
                }

                // chunk 2 (8 bytes):

                byte[] dataId = reader.ReadBytes(4);    // "data"
                int dataSize = reader.ReadInt32();      // size of audio data

                var ascii = Encoding.ASCII.GetString(dataId);

                //Debug.Assert(ascii == "data", "Data chunk not found!");

                while (ascii != "data")
                {
                    reader.ReadBytes(dataSize);

                    dataId = reader.ReadBytes(4);    // "data"
                    dataSize = reader.ReadInt32();      // size of audio data
                    ascii = Encoding.ASCII.GetString(dataId);
                }

                samplesPerSecond = sampleRate;                  // sample rate (usually 44100)
                samplesTotalCount = dataSize / (bitDepth / 8);  // total samples count in audio data

                // audio data:

                L = new double[samplesTotalCount / channels];

                for (int i = 0, s = 0; i < samplesTotalCount; i += 3)
                {
                    var b1 = reader.ReadByte();
                    var b2 = reader.ReadByte();
                    var b3 = reader.ReadByte();
                    int res = b1 | (b2 << 8) | (sbyte)b3 << 16;
                    L[s] = res / (double)(1 << (bitDepth - 1));
                    s++;
                }

                // metadata:

                long moreBytes = reader.BaseStream.Length - reader.BaseStream.Position;

                if (moreBytes > 0)
                {
                    metaData = reader.ReadBytes((int)moreBytes);
                }

                reader.Close();
            }
            catch (Exception e)
            {
                Debug.Fail("Failed to read file.");
                return false;
            }

            return true;
        }


        /// <summary>
        /// Writing all 16-bit stereo samples from arrays into wav file.
        /// </summary>

        public bool writeData(double[] L, double[] R)
        {
            Debug.Assert((samplesTotalCount != -1) && (samplesPerSecond != -1),
                "No sample count or sample rate info!");

            try
            {
                BinaryWriter writer = new BinaryWriter(File.Create(wavFilename));

                int fileSize = 44 + samplesTotalCount * 2;

                if (metaData != null)
                {
                    fileSize += metaData.Length;
                }

                // header:

                writer.Write(Encoding.ASCII.GetBytes("RIFF"));  // "RIFF"
                writer.Write((Int32)fileSize);                  // size of entire file with 16-bit data
                writer.Write(Encoding.ASCII.GetBytes("WAVE"));  // "WAVE"

                // chunk 1:

                writer.Write(Encoding.ASCII.GetBytes("fmt "));  // "fmt "
                writer.Write((Int32)16);                        // size of chunk in bytes
                writer.Write((Int16)1);                         // 1 - for PCM
                writer.Write((Int16)2);                         // only Stereo files in this version
                writer.Write((Int32)samplesPerSecond);          // sample rate per second (usually 44100)
                writer.Write((Int32)(4 * samplesPerSecond));    // bytes per second (usually 176400)
                writer.Write((Int16)4);                         // data align 4 bytes (2 bytes sample stereo)
                writer.Write((Int16)16);                        // only 16-bit in this version

                // chunk 2:

                writer.Write(Encoding.ASCII.GetBytes("data"));  // "data"
                writer.Write((Int32)(samplesTotalCount * 2));   // size of audio data 16-bit

                // audio data:

                for (int i = 0, s = 0; i < samplesTotalCount; i += 2)
                {
                    writer.Write(Convert.ToInt16(L[s]));
                    writer.Write(Convert.ToInt16(R[s]));
                    s++;
                }

                // metadata:

                if (metaData != null)
                {
                    writer.Write(metaData);
                }

                writer.Flush();
                writer.Close();
            }
            catch
            {
                Debug.Fail("Failed to write file.");
                return false;
            }

            return true;
        }
    }

}
