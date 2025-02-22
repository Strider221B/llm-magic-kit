# LLM Magic Kit

Terminal command line steps for uploading code as a dataset:  
rm -r ~/temp/kaggle_model_staging/llm_utilities/*  
cp -r ~/Git/kag_ai_olymp2025_01/kaggle_ai_olympiad_2025_01/models/ ~/temp/kaggle_model_staging/llm_utilities/models  
cp -r ~/Git/kag_ai_olymp2025_01/kaggle_ai_olympiad_2025_01/helpers/ ~/temp/kaggle_model_staging/llm_utilities/helpers  
cp -r ~/Git/kag_ai_olymp2025_01/kaggle_ai_olympiad_2025_01/dataset-metadata.json ~/temp/kaggle_model_staging/llm_utilities/dataset-metadata.json  
cd ~/temp/kaggle_model_staging/llm_utilities  
kaggle datasets version -p . -t -r zip -m "update"  
kaggle datasets create -p . -t -r zip  
