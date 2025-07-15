 commands : cd workspace 
git clone https://github.com/devasphn/NewTrialDia/ 
cd NewTrialDia
apt update 
apt-get install -y libsndfile1 ffmpeg build-essential git-lfs libcudnn8
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git
huggingface-cli login`.
python app.py
