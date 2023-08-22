import pandas as pd
from sklearn.cluster import KMeans

def perform_customer_segmentation(data):
  
  X = data[['Age', 'TotalPurchases', 'AveragePurchaseAmount']]
  
  kmeans = KMeans(n_clusters=5)
  kmeans.fit(X)
  
  data['Segment'] = kmeans.predict(X)

  return data


customer_data = pd.DataFrame({
  'CustomerID': [1, 2, 3, 4, 5], 
  'Age': [25, 34, 47, 58, 66],
  'Gender': ['F', 'M', 'F', 'M', 'F'],
  'TotalPurchases': [5, 7, 2, 10, 3],
  'AveragePurchaseAmount': [100, 60, 250, 50, 150]
})

segmented_data = perform_customer_segmentation(customer_data)

print(segmented_data)




#output:-
"""
   CustomerID  Age Gender  TotalPurchases  AveragePurchaseAmount  Segment
0           1   25      F               5                    100        3
1           2   34      M               7                     60        0
2           3   47      F               2                    250        1
3           4   58      M              10                     50        4
4           5   66      F               3                    150        2
"""
