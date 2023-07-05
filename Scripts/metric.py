def clipping_fn(logits, max_val, min_val):
  
  for i in range(len(logits)):
    for j in range(len(logits[i])):

        if logits[i][j] >= (max_val + min_val)/2:
          logits[i][j] = max_val
        else:
          logits[i][j] = min_val

  return (
      logits
  )

def metric_accuracy(logits, labels):
  confusion = [
      {
         'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 
      },
      {
         'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 
      },
      {
         'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 
      },
      {
         'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 
      },
  ]

  for i in range(len(logits)):
    for j in range(len(logits[i])):

      if logits[i][j] == 1 and labels[i][j] == 1:
          confusion[j]['tp'] += 1

      if logits[i][j] == 0 and labels[i][j] == 0:
          confusion[j]['tn'] += 1

      if logits[i][j] == 1 and labels[i][j] == 0:
          confusion[j]['fn'] += 1

      if logits[i][j] == 0 and labels[i][j] == 1:
          confusion[j]['fp'] += 1

  print(confusion)

  accuracy = list()

  for i in range(len(confusion)):
    accuracy.append((confusion[i]['tp'] + confusion[i]['tn'])/(confusion[i]['tp'] + confusion[i]['tn'] + confusion[i]['fp'] + confusion[i]['fn']))

  return (
      accuracy
  )
