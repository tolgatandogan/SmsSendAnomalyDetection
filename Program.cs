// <SnippetAddUsings>
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using System.Collections;
using System.Collections.Generic;
using System.Data;

// </SnippetAddUsings>

namespace SmsSendAnomalyDetection
{
    internal class Program
    {
        // <SnippetDeclareGlobalVariables>
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "sms-send.csv");

        // </SnippetDeclareGlobalVariables>
        private static void Main(string[] args)
        {
            // Create MLContext to be shared across the model creation workflow objects
            // <SnippetCreateMLContext>
            MLContext mlContext = new MLContext();
            // </SnippetCreateMLContext>

            //STEP 1: Common data loading configuration
            // <SnippetLoadData>
            IDataView dataView = mlContext.Data.LoadFromTextFile<SmsSendData>(path: _dataPath, hasHeader: true, separatorChar: ',');
            // </SnippetLoadData>

            // Detect seasonality for the series
            // <SnippetCallDetectPeriod>
            int period = DetectPeriod(mlContext, dataView);
            // </SnippetCallDetectPeriod>

            // Detect anomaly for the series with period information
            // <SnippetCallDetectAnomaly>
            DetectAnomaly(mlContext, dataView, period);
            // </SnippetCallDetectAnomaly>
        }

        private static int DetectPeriod(MLContext mlContext, IDataView phoneCalls)
        {
            Console.WriteLine("Detect period of the series");

            // STEP 2: Detect seasonality
            // <SnippetDetectSeasonality>
            int period = mlContext.AnomalyDetection.DetectSeasonality(phoneCalls, nameof(SmsSendData.value));
            // </SnippetDetectSeasonality>

            // <SnippetDisplayPeriod>
            Console.WriteLine("Period of the series is: {0}.", period);
            // </SnippetDisplayPeriod>

            return period;
        }

        private static void DetectAnomaly(MLContext mlContext, IDataView phoneCalls, int period)
        {
            Console.WriteLine("Detect anomaly points in the series");

            //STEP 2: Setup the parameters
            // <SnippetSetupSrCnnParameters>
            var options = new SrCnnEntireAnomalyDetectorOptions()
            {
                Threshold = 0.3,
                Sensitivity = 64.0,
                DetectMode = SrCnnDetectMode.AnomalyAndMargin,
                Period = period,
            };
            // </SnippetSetupSrCnnParameters>

            //STEP 3: Detect anomaly by SR-CNN algorithm
            // <SnippetDetectAnomaly>
            IDataView outputDataView =
                mlContext
                    .AnomalyDetection.DetectEntireAnomalyBySrCnn(
                        phoneCalls,
                        nameof(SmsSendPrediction.Prediction),
                        nameof(SmsSendData.value),
                        options);
            // </SnippetDetectAnomaly>

            // <SnippetCreateEnumerableForResult>
            IEnumerable<SmsSendPrediction> predictions = mlContext.Data.CreateEnumerable<SmsSendPrediction>(
                outputDataView, reuseRowObject: false);
            // </SnippetCreateEnumerableForResult>

            // <SnippetDisplayHeader>
            Console.WriteLine("Index\tAnomaly\tExpectedValue\tUpperBoundary\tLowerBoundary");
            // </SnippetDisplayHeader>

            // <SnippetDisplayAnomalyDetectionResults>
            var index = 0;

            foreach (var p in predictions)
            {
                if (p.Prediction[0] == 1)
                {
                    Console.WriteLine("{0},{1},{2},{3},{4}  <-- alert is on, detected anomaly", index,
                        p.Prediction[0], p.Prediction[3], p.Prediction[5], p.Prediction[6]);
                }
                else
                {
                    Console.WriteLine("{0},{1},{2},{3},{4}", index,
                        p.Prediction[0], p.Prediction[3], p.Prediction[5], p.Prediction[6]);
                }
                ++index;
            }

            Console.WriteLine("");
            // </SnippetDisplayAnomalyDetectionResults>
        }
    }
}