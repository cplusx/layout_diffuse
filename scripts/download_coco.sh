mkdir -p ~/disk2/data/COCO
cd ~/disk2/data/COCO
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
ls *.zip | while read f; do
        unzip $f;
done
