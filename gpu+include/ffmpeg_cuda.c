#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <cuda.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>

static AVFormatContext *ifmt = NULL;
static AVCodecContext  *dec  = NULL;
static AVBufferRef     *hwdev = NULL;
static int video_stream = -1;
static enum AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
static enum AVPixelFormat g_sw_format = AV_PIX_FMT_NV12;
static AVFrame *g_frame = NULL;


static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts){
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p){
        if (*p == AV_PIX_FMT_CUDA){ hw_pix_fmt = *p; return *p; }
    }
    return AV_PIX_FMT_NONE;
}

int fc_get_sw_format(void){ return (int)g_sw_format;}

int fc_open_cuda(const char *filename, int *out_w, int *out_h){
    int rc;
    rc = avformat_open_input(&ifmt, filename, NULL, NULL);
    rc = avformat_find_stream_info(ifmt, NULL);

    int vs = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    video_stream = vs;
    AVStream *st = ifmt->streams[video_stream];

    const AVCodec *codec = NULL;
    switch (st->codecpar->codec_id){
        case AV_CODEC_ID_AV1:
            codec = avcodec_find_decoder_by_name("av1_nvdec");
            if (!codec) codec = avcodec_find_decoder_by_name("av1_cuvid");
            break;
        case AV_CODEC_ID_HEVC: codec = avcodec_find_decoder_by_name("hevc_cuvid"); break;
        case AV_CODEC_ID_H264: codec = avcodec_find_decoder_by_name("h264_cuvid"); break;
        case AV_CODEC_ID_VP9:  codec = avcodec_find_decoder_by_name("vp9_cuvid");  break;
        default: break;
    }
    if (!codec) codec = avcodec_find_decoder(st->codecpar->codec_id);

    dec = avcodec_alloc_context3(codec);
    rc = avcodec_parameters_to_context(dec, st->codecpar);

    dec->thread_type  = FF_THREAD_FRAME;
    dec->thread_count = (st->codecpar->codec_id == AV_CODEC_ID_AV1) ? 1 : 0;

    rc = av_hwdevice_ctx_create(&hwdev, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0);

    dec->hw_device_ctx = av_buffer_ref(hwdev);
    dec->get_format = get_hw_format;

    g_sw_format = (st->codecpar->bits_per_raw_sample > 8) ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;

    AVBufferRef *frames_ref = av_hwframe_ctx_alloc(hwdev);
    AVHWFramesContext *frames = (AVHWFramesContext*)frames_ref->data;
    frames->format    = AV_PIX_FMT_CUDA;
    frames->sw_format = g_sw_format;
    frames->width     = st->codecpar->width;
    frames->height    = st->codecpar->height;
    frames->initial_pool_size = 4;

    if ((rc = av_hwframe_ctx_init(frames_ref)) < 0){
        av_buffer_unref(&frames_ref);
        return 0;
    }
    dec->hw_frames_ctx   = av_buffer_ref(frames_ref);
    dec->extra_hw_frames = 2;
    av_buffer_unref(&frames_ref);

    if ((rc = avcodec_open2(dec, codec, NULL)) < 0){
        return 0;
    }

    if (dec->pix_fmt != AV_PIX_FMT_CUDA){
        return 0;
    }

    *out_w = dec->width;
    *out_h = dec->height;

    if (!g_frame) g_frame = av_frame_alloc();

    printf("Decoder in use: %s  hw_pix_fmt=%d  sw_format=%d  %dx%d\n",
            dec->codec->name, (int)hw_pix_fmt, (int)g_sw_format, dec->width, dec->height);
    return 1;
}

int fc_read_cuda_luma(CUdeviceptr *d_luma, int *pitch, int *w, int *h){
    if (!ifmt || !dec) return 0;

    if (g_frame) av_frame_unref(g_frame);

    AVPacket pkt;
    for (;;){
        int rc = av_read_frame(ifmt, &pkt);
        if (rc == AVERROR_EOF) return 0;
        if (rc < 0) return 0;

        if (pkt.stream_index != video_stream){
            av_packet_unref(&pkt);
            continue;
        }

        rc = avcodec_send_packet(dec, &pkt);
        av_packet_unref(&pkt);
        if (rc < 0 && rc != AVERROR(EAGAIN)) return 0;

        for (;;){
            rc = avcodec_receive_frame(dec, g_frame);
            if (rc == AVERROR(EAGAIN)) break;
            if (rc == AVERROR_EOF)  return 0;
            if (rc < 0)             return 0;

            if (g_frame->format == hw_pix_fmt){
                *w     = g_frame->width;
                *h     = g_frame->height;
                *pitch = g_frame->linesize[0];
                *d_luma = (CUdeviceptr)(uintptr_t)g_frame->data[0];
                return 1;
            } else{
                av_frame_unref(g_frame);
            }
        }
    }
}

void fc_close(void){
    if (g_frame){ av_frame_free(&g_frame); g_frame = NULL; }

    if (dec){
        if (dec->hw_frames_ctx) av_buffer_unref(&dec->hw_frames_ctx);
        if (dec->hw_device_ctx) av_buffer_unref(&dec->hw_device_ctx);
        avcodec_free_context(&dec);
        dec = NULL;
    }
    if (ifmt){ avformat_close_input(&ifmt); ifmt = NULL; }
    if (hwdev){ av_buffer_unref(&hwdev); hwdev = NULL; }


}
