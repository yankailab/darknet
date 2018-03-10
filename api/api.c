#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>
#include "api.h"

static image g_imgBuf;
static image g_imgBufLetter;
static int g_nClass = 0;
static char** g_ppName = NULL;
static float** g_ppProb = NULL;
static box *g_pBox = NULL;
static network* g_pNet = NULL;
static float *g_pPrediction = NULL;
static int g_nDetect = 0;
static float* g_pAvg = NULL;

bool yoloInit(	const char* pCfgFile,
		const char* pWeightFile,
		const char* pLabelFile,
		int nClass,
		int nBatch)
{
    g_ppName = get_labels(pLabelFile);
    g_nClass = nClass;

    g_pNet = load_network(pCfgFile, pWeightFile, 0);
    set_batch_network(g_pNet, nBatch);

    layer L = g_pNet->layers[g_pNet->n-1];
    g_nDetect = L.n*L.w*L.h;
    g_pAvg = (float *) calloc(L.outputs, sizeof(float));

    g_pPrediction = (float *) calloc(L.outputs, sizeof(float));
    g_pBox = (box*) calloc(L.w*L.h*L.n, sizeof(box));
    g_ppProb = (float **) calloc(L.w*L.h*L.n, sizeof(float *));
    for(int i = 0; i < L.w*L.h*L.n; i++)
    {
	g_ppProb[i] = (float *)calloc(L.classes+1, sizeof(float));
    }

    g_imgBuf.data = NULL;

    return true;
}

int yoloUpdate(IplImage* pImg,
		yolo_object* pObj,
		int nDetect,
		float thresh,
		float hier,
		float nms)
{
    if (!pImg)return -1;
    if (!pObj)return -1;
    if (nDetect <= 0)return -1;

    if(g_imgBuf.data)
    {
	ipl_into_image(pImg, g_imgBuf);
	rgbgr_image(g_imgBuf);
	letterbox_image_into(g_imgBuf, g_pNet->w, g_pNet->h, g_imgBufLetter);
    }
    else
    {
	g_imgBuf = ipl_to_image(pImg);
	rgbgr_image(g_imgBuf);
	g_imgBufLetter = letterbox_image(g_imgBuf, g_pNet->w, g_pNet->h);
    }

    layer L = g_pNet->layers[g_pNet->n-1];
    float *pPred = network_predict(g_pNet, g_imgBufLetter.data);

    memcpy(g_pPrediction, pPred, L.outputs*sizeof(float));

    if(L.type == DETECTION)
    {
        get_detection_boxes(L, 1, 1, thresh, g_ppProb, g_pBox, 0);
    }
    else if (L.type == REGION)
    {
        get_region_boxes(L, g_imgBuf.w, g_imgBuf.h, g_pNet->w, g_pNet->h, thresh, g_ppProb, g_pBox, 0, 0, 0, hier, 1);
    }

    if (nms > 0) do_nms_obj(g_pBox, g_ppProb, L.w*L.h*L.n, L.classes, nms);

    int iDetect = 0;
    int i,j;
    for (i = 0; i < g_nDetect; i++)
    {
	yolo_object* pO = &pObj[iDetect];
	pO->m_iClass = -1;
	pO->m_prob = 0.0;	

        for (j = 0; j < g_nClass; j++)
	{
	    float prob = g_ppProb[i][j];
	    if (prob < thresh)continue;

            if (prob > pO->m_prob)
	    {
            	pO->m_iClass = j;
		pO->m_pName = g_ppName[j];
		pO->m_prob = g_ppProb[i][j];
            }
        }

	if(pO->m_iClass < 0) continue;

        box b = g_pBox[i];
	b.w *= 0.5;
	b.h *= 0.5;
        pO->m_l = b.x - b.w;
        pO->m_r = b.x + b.w;
        pO->m_t = b.y - b.h;
        pO->m_b = b.y + b.h;

	if(++iDetect >= nDetect) return nDetect;
    }

    return iDetect;
}

