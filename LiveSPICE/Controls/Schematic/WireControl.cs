﻿using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Xml.Linq;

namespace LiveSPICE
{
    public class WireControl : ElementControl
    {
        static WireControl()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(WireControl), new FrameworkPropertyMetadata(typeof(WireControl)));
        }

        protected Circuit.Wire wire;
        public WireControl(Circuit.Wire W) : base(W) 
        { 
            wire = W;
            MouseMove += (s, e) => ToolTip = wire.Node != null ? wire.Node.ToString() : null;
        }
        
        protected override void OnRender(DrawingContext dc)
        {
            dc.PushGuidelineSet(Guidelines);

            // This isn't pointless, it makes WPF mouse hit tests succeed near the wire instead of exactly on it.
            dc.DrawRectangle(Brushes.Transparent, null, new Rect(-2, -2, ActualWidth + 4, ActualHeight + 4));

            if (Selected)
                dc.DrawLine(SelectedWirePen, new Point(0, 0), new Point(ActualWidth, ActualHeight));
            else if (Highlighted)
                dc.DrawLine(HighlightedWirePen, new Point(0, 0), new Point(ActualWidth, ActualHeight));
            else
                dc.DrawLine(WirePen, new Point(0, 0), new Point(ActualWidth, ActualHeight));
            
            DrawTerminal(dc, ToPoint(wire.A - wire.LowerBound), wire.Anode.ConnectedTo != null);
            DrawTerminal(dc, ToPoint(wire.B - wire.LowerBound), wire.Cathode.ConnectedTo != null);

            dc.Pop();
        }
        
        protected static Pen SelectedWirePen = new Pen(Brushes.Blue, EdgeThickness) { StartLineCap = PenLineCap.Round, EndLineCap = PenLineCap.Round };
        protected static Pen HighlightedWirePen = new Pen(Brushes.Gray, EdgeThickness) { StartLineCap = PenLineCap.Round, EndLineCap = PenLineCap.Round };
    }
}
