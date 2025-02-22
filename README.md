# LLM Magic Kit

Terminal command line steps for uploading code as a dataset:  
  
rm -r ~/temp/kaggle_model_staging/llm_utilities/*  
cp -r ~/Git/llm-magic-kit/models/ ~/temp/kaggle_model_staging/llm_utilities/models  
cp -r ~/Git/llm-magic-kit/helpers/ ~/temp/kaggle_model_staging/llm_utilities/helpers  
cp -r ~/Git/llm-magic-kit/dataset-metadata.json ~/temp/kaggle_model_staging/llm_utilities/dataset-metadata.json  
cd ~/temp/kaggle_model_staging/llm_utilities  
kaggle datasets version -p . -t -r zip -m "update"  
  
kaggle datasets create -p . -t -r zip  
  
The requirements.txt lists down all the libraries installed on Kaggle on Feb 22, 2025 18:22 IST (UTC +5:30) as the code was primarily verified on Kaggle.  
