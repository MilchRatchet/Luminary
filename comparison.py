import flip_evaluator as flip
import matplotlib.pyplot as plt
import argparse
import glob

def GatherData(ref, testPath):
  if testPath == None:
    return [], [], []

  files = glob.glob(testPath[0] + '/*.png')
  resultsFile = testPath[0] + '/LuminaryBenchmarkResults.txt'

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

  return errors, sampleCounts, times


if __name__ == '__main__':
  parser = argparse.ArgumentParser("simple_example")
  parser.add_argument("--reference", type=str, nargs=1, required=True, help="Path to reference image")
  parser.add_argument("--testA", type=str, nargs=1, required=True, help="Path to directory containing test images of run A")
  parser.add_argument("--testB", type=str, nargs=1, required=False, help="Path to directory containing test images of run B")
  parser.add_argument("--output", type=str, nargs=1, required=True, help="Path to output directory")
  args = parser.parse_args()

  ref = args.reference[0]

  errorsA, sampleCountsA, timesA = GatherData(ref, args.testA)
  errorsB, sampleCountsB, timesB = GatherData(ref, args.testB)

  plt.plot(sampleCountsA, errorsA, color='green')
  plt.plot(sampleCountsB, errorsB, color='orange')

  plt.title('Convergence (Number of Samples)')
  plt.ylabel('Mean Error')
  plt.xlabel('Number of Samples')

  axes = plt.gca()

  axes.set_xscale('log')

  plt.savefig(args.output[0] + "/error_samples.png")

  plt.figure()

  plt.plot(timesA, errorsA, color='green')
  plt.plot(timesB, errorsB, color='orange')

  plt.title('Convergence (Execution Time)')
  plt.ylabel('Mean Error')
  plt.xlabel('Execution Time')

  plt.savefig(args.output[0] + "/error_times.png")

