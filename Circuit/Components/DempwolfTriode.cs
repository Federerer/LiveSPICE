using ComputerAlgebra;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Circuit.Components
{
    [Category("Vacuum Tubes")]
    [DisplayName("Triode (Dempwolf)")]
    [Description("Triode implemented using Dempwolf's model.")]
    public class DempwolfTriode : Triode
    {
        private double g = 2.242E-3;
        private double mu = 103.2;
        private double gamma = 1.26;
        private double c = 3.4;

        private double gg = 6.177E-4;
        private double xi = 1.314;
        private double cg = 9.901;
        private Quantity ig0 = new Quantity(8.025E-8, Units.A);

        [Serialize]
        public double Mu { get { return mu; } set { mu = value; NotifyChanged(nameof(Mu)); } }

        [Serialize]
        public double Gamma { get { return gamma; } set { gamma = value; NotifyChanged(nameof(Gamma)); } }

        [Serialize]
        public double G { get { return g; } set { g = value; NotifyChanged(nameof(G)); } }

        [Serialize]
        public double Gg { get { return gg; } set { gg = value; NotifyChanged(nameof(Gg)); } }

        [Serialize]
        public double C { get { return c; } set { c = value; NotifyChanged(nameof(C)); } }
        
        [Serialize]
        public double Cg { get { return cg; } set { cg = value; NotifyChanged(nameof(Cg)); } }
        
        [Serialize]
        public double Xi { get { return xi; } set { xi = value; NotifyChanged(nameof(Xi)); } }
        
        [Serialize]
        public Quantity Ig0 { get { return ig0; } set { ig0 = value; NotifyChanged(nameof(Ig0)); } }

        protected override void Analyze(Analysis Mna, Expression Vgk, Expression Vpk, out Expression ip, out Expression ig)
        {
            var tmp = Cg * Vgk;
            ig = Call.If(Vgk > -5, Gg * Binary.Power(Call.If(tmp < 5, Call.Ln(1 + LinExp(tmp)), tmp) / Cg, Xi), 0) + Ig0;

            Expression ex = C * ((Vpk / Mu) + Vgk);
            var ik =  Call.If(ex > -5, G * Binary.Power(Call.If(ex < 5, Call.Ln(1 + LinExp(ex)), ex) / C, Gamma), 0);

            ip = ik - ig;
        }
    }
}
