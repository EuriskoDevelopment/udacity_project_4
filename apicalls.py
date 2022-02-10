import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"



#Call each API endpoint and store the responses
response1 = requests.get(URL+'/prediction?filename=testdata/testdata.csv').content.decode('utf8')
response2 = requests.get(URL+'/scoring?filename=testdata.csv').content.decode('utf8')
response3 = requests.get(URL+'/summarystats?filename=finaldata.csv').content.decode('utf8')
response4 = requests.get(URL+'/diagnostics').content.decode('utf8')

#combine all API responses
responses = "\n".join([response1, response2, response3, response4])

#write the responses to your workspace
print(responses)


