import flip_evaluator as flip
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import os

def GatherData(ref, testPath):
  files = glob.glob(testPath + '/*.png')
  resultsFile = testPath + '/LuminaryBenchmarkResults.txt'

  errors = []
  for img in files:
    flipErrorMap, meanFLIPError, parameters = flip.evaluate(ref, img, "LDR")
    print(img + " Error: " + str(meanFLIPError))
    errors.append(meanFLIPError)

  sampleCounts = []
  times = []
  with open(resultsFile) as f:
    lines = f.read().splitlines()
    for line in lines:
      values = line.split(",")
      sampleCounts.append(int(values[0]))
      times.append(float(values[1]))

  return (os.path.basename(testPath), errors, sampleCounts, times)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("simple_example")
  parser.add_argument("--reference", type=str, nargs=1, required=True, help="Path to reference image")
  parser.add_argument("--test", type=str, nargs='+', required=True, help="Paths to directories containing test images")
  parser.add_argument("--output", type=str, nargs=1, required=True, help="Path to output directory")
  args = parser.parse_args()

  ref = args.reference[0]

  results = []
  for testSet in args.test:
    results.append(GatherData(ref, testSet))


  maxSampleCount = 1
  for result in results:
    maxSampleCount = max(maxSampleCount, max(result[2]))

  axes = plt.gca()

  axes.set_yscale('log')

  for result in results:
    plt.plot(result[2], result[1], label=result[0], linestyle='dashed', linewidth=0.5)

  plt.title('Convergence (Number of Samples)')
  plt.ylabel('Mean FLIP Error')
  plt.xlabel('Number of Samples')
  plt.legend(loc='best')
  plt.xticks(np.linspace(0, maxSampleCount, 5))

  plt.savefig(args.output[0] + "/error_samples.png", format='png', dpi=1200)

  plt.figure()

  axes = plt.gca()

  axes.set_yscale('log')

  for result in results:
    plt.plot(result[3], result[1], label=result[0], linestyle='dashed', linewidth=0.5)

  plt.title('Convergence (Execution Time)')
  plt.ylabel('Mean FLIP Error')
  plt.xlabel('Execution Time (s)')
  plt.legend(loc='best')

  plt.savefig(args.output[0] + "/error_times.png", format='png', dpi=1200)

