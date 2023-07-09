wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xvf ffmpeg-release-amd64-static.tar.xz
mkdir -p bin && mv ffmpeg-release-amd64-static/ffmpeg ./bin

export PATH=/home/ec2-user/ffmpeg/bin:$PATH
python run.py -s ./image/song1.jpg -t ./video/2.mp4 -o ./output/b2.mp4 --execution-provider cuda --keep-fps --frame-processor face_swapper --keep-audio
