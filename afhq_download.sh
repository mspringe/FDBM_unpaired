URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
ZIP_FILE=./afhq.zip
mkdir -p ./datasets/AFHQ
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./datasets/AFHQ
mv ./datasets/AFHQ/afhq/* ./datasets/AFHQ/
rmdir ./datasets/AFHQ/afhq/
rm $ZIP_FILE
