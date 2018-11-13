docker image build -t tensor1.2_cuda8.0 /tf-my-faster-rcnn-simple/docker/
docker image ls
docker container ls --all
#new container
docker container run -p 8000:3000 -it tensor1.2_cuda8.0 /bin/bash
#start container
docker container start 424603ff6159
#entry container
docker attach 2a2adbe25c7d
#kill 
docker container kill 2a2adbe25c7d
#copy file to container
docker cp ./tools/demo.py 2a2adbe25c7d:/root/chenxingli/my-tf-faster-rcnn-simple/tools/demo.py
#copy file from container
docker cp 2a2adbe25c7d:/root/chenxingli/my-tf-faster-rcnn-simple/tools /home/chenxingli/my-tf-faster-rcnn-simple/tools
#run 
python ./tools/demo.py --net=vgg16 --dataset=pascal_voc
#�������ļ����ڵ�����,��ָ�������Ļ��ͻ�������һ������
sudo docker run -it -v /home/dengta/faster-rcnn/:/root/chenxingli/ fasterrcnn /bin/bash
#��-v�����������ݾ�������
docker run -v /home/chenxingli/:/root/chenxingli/ $5f756132ac9d


