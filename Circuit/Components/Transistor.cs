﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SyMath;
using System.ComponentModel;

namespace Circuit
{
    [TypeConverter(typeof(ModelConverter<BJTModel>))]
    public abstract class BJTModel : Model
    {
        public BJTModel(string Name) : base(Name) { }

        public abstract void Evaluate(Expression Vbc, Expression Vbe, out Expression ic, out Expression ib, out Expression ie);

        public static List<BJTModel> Models { get { return Model.GetModels<BJTModel>(); } }

        static BJTModel()
        {
            Models.Add(new EbersMollModel("2SC2240", 1e-12, 200, 0.1));
        }
    }

    // http://people.seas.harvard.edu/~jones/es154/lectures/lecture_3/bjt_models/ebers_moll/ebers_moll.html
    public class EbersMollModel : BJTModel
    {
        private double bf;
        private double br;
        private double _is;

        public double BF { get { return bf; } set { bf = value; } }
        public double BR { get { return br; } set { br = value; } }
        public double IS { get { return _is; } set { _is = value; } }
        
        public EbersMollModel(string Name, double IS, double BF, double BR) : base(Name)
        {
            bf = BF;
            br = BR;
            _is = IS;
        }

        public override void Evaluate(Expression Vbc, Expression Vbe, out Expression ic, out Expression ib, out Expression ie)
        {
            double aR = BR / (1 + BR);
            double aF = BF / (1 + BF);

            Expression iDE = IS * (Call.Exp(Vbe / Component.VT) - 1);
            Expression iDC = IS * (Call.Exp(Vbc / Component.VT) - 1);

            ie = iDE - aR * iDC;
            ic = -iDC + aF * iDE;
            ib = (1 - aF) * iDE + (1 - aR) * iDC;
        }

        public override string ToString() { return base.ToString() + " (Ebers-Moll)"; }
    };
    
    /// <summary>
    /// Transistors.
    /// </summary>
    [CategoryAttribute("Transistors")]
    [DisplayName("BJT")]
    public class BJT : Component
    {
        protected Terminal c, e, b;
        public override IEnumerable<Terminal> Terminals 
        { 
            get 
            {
                yield return c;
                yield return e;
                yield return b;
            } 
        }
        [Browsable(false)]
        public Terminal Collector { get { return c; } }
        [Browsable(false)]
        public Terminal Emitter { get { return e; } }
        [Browsable(false)]
        public Terminal Base { get { return b; } }

        protected BJTModel model = BJTModel.Models.First();
        [Serialize]
        public BJTModel Model { get { return model; } set { model = value; NotifyChanged("Model"); } }

        public BJT()
        {
            c = new Terminal(this, "C");
            e = new Terminal(this, "E");
            b = new Terminal(this, "B");
            Name = "Q1";
        }

        public override void Analyze(ModifiedNodalAnalysis Mna)
        {
            Expression Vbc = Mna.AddNewUnknownEqualTo(Name + "bc", b.V - c.V);
            Expression Vbe = Mna.AddNewUnknownEqualTo(Name + "be", b.V - e.V);

            Expression ic, ib, ie;
            model.Evaluate(Vbc, Vbe, out ic, out ib, out ie);
            ic = Mna.AddNewUnknownEqualTo("i" + Name + "c", ic);
            ib = Mna.AddNewUnknownEqualTo("i" + Name + "b", ib);
            Mna.AddTerminal(c, ic);
            Mna.AddTerminal(b, ib);
            Mna.AddTerminal(e, -(ic + ib));
        }

        public override void LayoutSymbol(SymbolLayout Sym)
        {
            Sym.AddTerminal(c, new Coord(10, 20));
            Sym.AddTerminal(b, new Coord(-20, 0));
            Sym.AddTerminal(e, new Coord(10, -20));

            int bx = -5;
            Sym.AddWire(c, new Coord(10, 17));
            Sym.AddWire(b, new Coord(bx, 0));
            Sym.AddWire(e, new Coord(10, -17));

            Sym.DrawLine(EdgeType.Black, new Coord(bx, 12), new Coord(bx, -12));
            Sym.DrawLine(EdgeType.Black, new Coord(10, 17), new Coord(bx, 8));
            Sym.DrawArrow(EdgeType.Black, new Coord(bx, -8), new Coord(10, -17), 0.2, 0.3);

            Sym.DrawText(Model.Name, new Coord(8, 20), Alignment.Far, Alignment.Near);
            Sym.DrawText(Name, new Point(8, -20), Alignment.Far, Alignment.Far);

            Sym.AddCircle(EdgeType.Black, new Coord(0, 0), 20);
        }
    }
}
