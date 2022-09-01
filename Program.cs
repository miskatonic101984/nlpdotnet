using System;
using System.Text.Json;

namespace BertMlNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var directory = Directory.GetCurrentDirectory();
            var vocabBase = "Assets\\Vocabulary\\vocab.txt";
            var modelBase =  "Assets\\Models\\bertsquad-10.onnx";
            var vocabPath = Path.Combine(directory,vocabBase);
            var modelPath = Path.Combine(directory,modelBase);
            var model = new Bert(vocabPath,modelPath);                  

            var (tokens, probability) = model.Predict(args[0], args[1]);

            Console.WriteLine(JsonSerializer.Serialize(new
            {
                Probability = probability,
                Tokens = tokens
            }));
        }
    }
}