#ifndef DARKNET_API_H
#define DARKNET_API_H

typedef struct yolo_object
{
    int m_iClass;
    float m_prob;
    char* m_pName;
    float m_l;
    float m_t;
    float m_r;
    float m_b;
}yolo_object;

bool yoloInit( const char* pCfgFile,	//"/home/kai/dev/darknet/cfg/yolo.cfg"
		const char* pWeightFile,		//"/home/kai/dev/darknet/data/yolo.weights"
		const char* pLabelFile,		//"/home/kai/dev/darknet/data/names.list"
		int nClass,				//80
		int nBatch);			//1

int yoloUpdate( IplImage* pImg,
		yolo_object* pObj,
		int nDetect,
		float thresh,		//0.24
		float hier,			//0.5
		float nms); 		//0.4

#endif

