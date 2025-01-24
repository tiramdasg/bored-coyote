import time

import requests

for idx in range(1000):
    time.sleep(10)

    my_str1 = "Trump"
    my_str2 = "Biden"

    my_str3 = "GOP"
    my_str4 = "Republicans"
    my_str5 = "Democrats"
    resp1 = requests.post("http://localhost:8082/predict_keywords", json={"text": my_str1})
    resp2 = requests.post("http://localhost:8082/predict_keywords", json={"text": my_str2})
    resp3 = requests.post("http://localhost:8082/predict_keywords", json={"text": my_str3})
    resp4 = requests.post("http://localhost:8082/predict_keywords", json={"text": my_str4})
    resp5 = requests.post("http://localhost:8082/predict_keywords", json={"text": my_str5})

    print(my_str1 + ":", resp1.json(),
          my_str2 + ":", resp2.json(),
          my_str3 + ":", resp3.json(),
          my_str4 + ":", resp4.json(),
          my_str5 + ":", resp5.json())

    # Trump: [0.49214] Biden: [0.99474] GOP: [0.00035] Republicans: [0.01058] Democrats: [0.98659]
    # Trump: [0.52483] Biden: [0.97316] GOP: [0.00118] Republicans: [0.00828] Democrats: [0.99897]