



API calls for generating the .txt in practicemodels:
curl 127.0.0.1:8000/prediction?filename='testdata/testdata.csv' > practicemodels/api_predictions.txt
curl 127.0.0.1:8000/scoring?filename='testdata.csv' > practicemodel/api_scoring.txt
curl 127.0.0.1:8000/summarystats?filename='finaldata.csv' > practicemodels/api_summarystats.txt
curl 127.0.0.1:8000/diagnostics > practicemodels/api_diagnostics.txt
