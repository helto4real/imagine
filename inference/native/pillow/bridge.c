/* Ownership/error/GIL hooks only; resampling and conversion bodies are upstream. */
#include "Imaging.h"
#include <pthread.h>
#include <stdio.h>
static pthread_mutex_t gate = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local char error_message[256];
static _Thread_local int error_kind;
void imagine_pillow_error_clear(void) { error_message[0]=0;error_kind=0; }
void imagine_pillow_arrow_unavailable(void) { abort(); }
void *ImagingError_ValueError(const char *message) { error_kind=1;snprintf(error_message,sizeof(error_message),"%s",message);return NULL; }
void *ImagingError_MemoryError(void) { error_kind=2;error_message[0]=0;return NULL; }
void *ImagingError_ModeError(void) { return ImagingError_ValueError("image has wrong mode"); }
void *ImagingError_Mismatch(void) { return ImagingError_ValueError("images do not match"); }
/* These sections release/reacquire the original Python GIL. The standalone
   buffer owner supplies the equivalent allocator gate, preserving parallel C
   compute inside sections while serializing libImaging's shared arena access. */
void ImagingSectionEnter(ImagingSectionCookie *cookie) { *cookie=NULL;pthread_mutex_unlock(&gate); }
void ImagingSectionLeave(ImagingSectionCookie *cookie) { (void)cookie;pthread_mutex_lock(&gate); }
__attribute__((visibility("default")))
const char *imagine_pillow_error(void) { return error_message; }
__attribute__((visibility("default")))
int imagine_pillow_abi(void) { return 2; }
/* PIL.Image.resize filters by their Resampling value, as Image.resize maps them. */
static int transform(int filter) {
    switch(filter){
    case 1: return IMAGING_TRANSFORM_LANCZOS;
    case 2: return IMAGING_TRANSFORM_BILINEAR;
    case 3: return IMAGING_TRANSFORM_BICUBIC;
    case 4: return IMAGING_TRANSFORM_BOX;
    case 5: return IMAGING_TRANSFORM_HAMMING;
    default: return -1;
    }
}
__attribute__((visibility("default")))
int imagine_pillow_resize_u8_filter(const unsigned char *src,int w,int h,int channels,int ow,int oh,int filter,unsigned char *dst) {
    Imaging image=NULL,premult=NULL,resized=NULL,converted=NULL;
    pthread_mutex_lock(&gate);imagine_pillow_error_clear();
    if (w<=0||h<=0||ow<=0||oh<=0) { ImagingError_ValueError("height and width must be > 0");goto finish; }
    if (channels!=3&&channels!=4) { ImagingError_ValueError("expected RGB or RGBA buffer");goto finish; }
    /* Nearest goes through ImagingTransform in the original; it is not bound here. */
    if (transform(filter)<0) { ImagingError_ValueError("unsupported resampling filter");goto finish; }
    image=ImagingNew(channels==4?IMAGING_MODE_RGBA:IMAGING_MODE_RGB,w,h);if(!image)goto finish;
    for(int y=0;y<h;y++)for(int x=0;x<w;x++){
        unsigned char *p=(unsigned char*)image->image[y]+4*x;
        for(int c=0;c<channels;c++)p[c]=src[((size_t)y*w+x)*channels+c];
    }
    if(w==ow&&h==oh) { memcpy(dst,src,(size_t)w*h*channels);goto finish; }
    Imaging input=image;
    if(channels==4){premult=ImagingConvert(image,IMAGING_MODE_RGBa,NULL,3);if(!premult)goto finish;input=premult;}
    float box[4]={0,0,(float)w,(float)h};
    resized=ImagingResample(input,ow,oh,transform(filter),box);if(!resized)goto finish;
    Imaging output=resized;
    if(channels==4){converted=ImagingConvert(resized,IMAGING_MODE_RGBA,NULL,3);if(!converted)goto finish;output=converted;}
    for(int y=0;y<oh;y++)for(int x=0;x<ow;x++)for(int c=0;c<channels;c++)dst[((size_t)y*ow+x)*channels+c]=((unsigned char*)output->image[y])[4*x+c];
finish:
    ImagingDelete(converted);ImagingDelete(resized);ImagingDelete(premult);ImagingDelete(image);
    int result=error_kind;pthread_mutex_unlock(&gate);return result;
}
__attribute__((visibility("default")))
int imagine_pillow_resize_u8(const unsigned char *src,int w,int h,int channels,int ow,int oh,unsigned char *dst) {
    return imagine_pillow_resize_u8_filter(src,w,h,channels,ow,oh,1,dst);
}
