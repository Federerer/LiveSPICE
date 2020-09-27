using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace SignalProcessing
{
    public class FIRFilter
    {
        private int _filterLength;
        private double[] delayLine;
        private double[] impulseResponse;
        private int count = 0;
        private int _simdLen;

        public FIRFilter(double[] coefs)
        {
            _simdLen = Vector<double>.Count;

            _filterLength = 620; //coefs.Length + (_simdLen - (coefs.Length % _simdLen));
            impulseResponse = new double[_filterLength];
            System.Array.Copy(coefs.Take(_filterLength).Reverse().ToArray(), 0, impulseResponse, 0, Math.Min(coefs.Length, _filterLength));
            delayLine = new double[_filterLength*2];
        }

        public void ProcessSamples(double[] inputSample)
        {
            for (int i = 0; i < inputSample.Length; i++)
            {
                delayLine[count] = delayLine[count + _filterLength] = inputSample[i];

                count++;
                if((count + 1) > _filterLength)
                    count = 0;

                var res = 0.0;

                for (int j = 0; j < _filterLength; j += _simdLen)
                {
                    var impulseVector = new Vector<double>(impulseResponse, j);
                    var samples = new Vector<double>(delayLine, (count + j));
                    res += Vector.Dot(samples, impulseVector);
                }

                inputSample[i] = res;
            }
        }
    }
}
