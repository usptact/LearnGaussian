using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;

namespace LearnGaussianWithMissingData
{
    class Program
    {
        static void Main(string[] args)
        {
            Rand.Restart(5209);     // www.random.org
            Random rnd = new Random(398);   // ww.random.org

            // data points
            double[] data = new double[] { -1, 0.1, 0.3, -1, -0.2, 0.05, -0.05, -1, 0.05, 0.01 };
            bool[] isMissing = new bool[] { true, false, false, true, false, false, false, true, false, false };
            
            Range r = new Range(data.Length);

            // create mean & precision RVs
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0.0, 10.0).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1.0, 1.0).Named("precision");

            // create observed RV array for data
            VariableArray<double> x = Variable.Array<double>(r);
            //x[r] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(r);
            x.ObservedValue = data;

            // create missing data indicator RV array (observed)
            VariableArray<bool> missing = Variable.Observed(isMissing, r);

            using (Variable.ForEach(r))
            {
                using (Variable.IfNot(missing[r]))
                {
                    x[r] = Variable.GaussianFromMeanAndPrecision(mean, precision);
                }
            }

            InferenceEngine engine = new InferenceEngine();

            Console.WriteLine( "mean posterior = " + engine.Infer(mean) );
            Console.WriteLine( "precision posterior = " + engine.Infer(precision) );

            Console.WriteLine("\nPress any key...");
            Console.ReadLine();
        }
    }
}
