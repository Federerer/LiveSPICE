using System;
using System.Linq;
using Circuit;
using ComputerAlgebra;
using ComputerAlgebra.LinqCompiler;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests
{
    [TestClass]
    public class UnitTest1
    {

        public double Mu { get; set; } = 100.0;
        public double Ex { get; set; } = 1.4;
        public double Kp { get; set; } = 600;
        public double Kvb { get; set; } = 300;

        private double Kg = 1060.0;
        private Quantity Rgk = new Quantity(1e6, Units.Ohm);
        private Quantity Vg = new Quantity(0.33, Units.V);



        [DataTestMethod]
        [DataRow(50)]
        [DataRow(1000)]
        public void TestMethod1(double input)
        {
            Expression Vgk = "Vgk";
            Expression Vpk = "Vpk";

            Expression ex = Kp * (1.0 / Mu + Vgk * Binary.Power((Kvb + Vpk * Vpk), -0.5));



            Expression E1 = Call.If(ex > 5, ex, Call.If(ex < -5,0, Call.Ln(1 + Component.LinExp(ex)))) * Vpk / Kp;
            Expression E2 = Call.Ln(1 + Component.LinExp(ex)) * Vpk / Kp;


            var Ip = Call.If(E1 > 0, (E1 ^ Ex) / Kg, 0);
            var Ig = Call.If(Vgk > Vg, (Vgk - Vg) / Rgk, 0);

            var ip = Ip.Evaluate().Compile<Func<double,double, double>>("Vgk", "Vpk");

            var res = Enumerable.Range(1, 1000).Select(i => Math.Sin(i / (1000 / Math.PI)));

            Assert.AreEqual(E1, E2);

        }
    }
}
