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


//./darknet detector demo cfg/coco.data cfg/yolo.cfg data/yolo.weights -c 1


static image g_imgBuf;
static g_pIPLImage* g_pIPL;

static char **g_ppName;
static int g_nClass;
static float **g_ppProb;
static box *g_pBox;
static network *g_pNet;
static float **g_ppPrediction;
static float g_thresh = 0;
static float g_hier = .5;
static int g_nDetect = 0;
static float *g_pAvg;

bool yoloInit(	char* pDataFile, 		//"/home/kai/dev/darknet/cfg/coco.data"
		char* pCfgFile, 		//"/home/kai/dev/darknet/cfg/yolo.cfg"
		char* pWeightFile,		//"/home/kai/dev/darknet/data/yolo.weights"
		char* pLabelFile,		//"/home/kai/dev/darknet/data/names.list"
		float thresh,			//0.24
		float hier,			//0.5
		int w,
		int h,
		int nChannel,
		int nBatch)
{

    list *pOpt = read_data_cfg(pDataFile);
    g_nClass = option_find_int(pOpt, "classes", 20);
    g_ppName = get_labels(pLabelFile);
    g_ppPrediction = calloc(demo_frame, sizeof(float*));
    g_thresh = thresh;
    g_hier = hier;

    g_pNet = load_network(pCfgfile, pWeightFile, 0);
    set_batch_network(g_pNet, 1);

    layer L = g_pNet->layers[g_pNet->n-1];
    g_nDetect = L.n*L.w*L.h;
    g_pAvg = (float *) calloc(L.outputs, sizeof(float));

    int j;
    for(j = 0; j < demo_frame; ++j) g_ppPrediction[j] = (float *) calloc(L.outputs, sizeof(float));

    g_pBox = (box*)calloc(L.w*L.h*L.n, sizeof(box));
    g_ppProb = (float **)calloc(L.w*L.h*L.n, sizeof(float *));
    for(j = 0; j < L.w*L.h*L.n; ++j) g_ppProb[j] = (float *)calloc(L.nClass+1, sizeof(float));

    g_pIPL = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, nChannel);

    return true;
}


void yoloUpdate(Mat* pImg)
{
    int status = fill_image_from_stream(cap, g_imgBuf);

    float nms = .4;

    layer L = g_pNet->layers[g_pNet->n-1];
    float *X = g_imgBuf_letter[(g_imgBuf_index+2)%3].data;
    float *prediction = network_predict(g_pNet, X);

    memcpy(g_ppPrediction[demo_index], prediction, L.outputs*sizeof(float));
    mean_arrays(g_ppPrediction, demo_frame, L.outputs, g_pAvg);
    L.output = g_pAvg;

    if(L.type == DETECTION)
    {
        get_detection_g_pBox(l, 1, 1, g_thresh, g_ppProb, g_pBox, 0);
    }
    else if (L.type == REGION)
    {
        get_region_g_pBox(l, g_imgBuf[0].w, g_imgBuf[0].h, g_pNet->w, g_pNet->h, g_thresh, g_ppProb, g_pBox, 0, 0, 0, g_hier, 1);
    }
    else
    {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(g_pBox, g_ppProb, L.w*L.h*L.n, L.nClass, nms);

    image display = g_imgBuf[(g_imgBuf_index+2) % 3];
    draw_detections(display, g_nDetect, g_thresh, g_pBox, g_ppProb, 0, g_ppName, demo_alphabet, g_nClass);

}

