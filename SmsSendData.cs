﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SmsSendAnomalyDetection
{
    // <SnippetDeclareTypes>
    public class SmsSendData
    {
        [LoadColumn(0)]
        public string timestamp;

        [LoadColumn(1)]
        public double value;
    }

    public class SmsSendPrediction
    {
        //vector to hold anomaly detection results. Including isAnomaly, anomalyScore, magnitude, expectedValue, boundaryUnits, upperBoundary and lowerBoundary.
        [VectorType(7)]
        public double[] Prediction { get; set; }
    }

    // </SnippetDeclareTypes>
}