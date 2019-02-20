#ifndef __CNN_TEXT_H__
#define __CNN_TEXT_H__

static inline int cnn_text_get_index(int wShift, int hShift, int w, int h,
                                     int ch, int width, int height, int channel)
{
    int imSize;
    int row, col;

    // Get size
    imSize = width * height;

    // Find row, col index
    row = h + hShift;
    if (row < 0)
    {
        row = 0;
    }
    if (row >= height)
    {
        row = height;
    }

    col = w + wShift;
    if (col < 0)
    {
        col = 0;
    }
    if (col >= width)
    {
        col = width;
    }

    // Get index
    return ch * imSize + row * width + col;
}

static inline void cnn_text_unroll(int* nbrMap, int* ctrMap, int width,
                                   int height, int channel)
{
    const int wSize = 8;
    int mapCols = channel * wSize;

    for (int h = 0; h < height; h++)
    {
        int rowBase = h * width;

        for (int w = 0; w < width; w++)
        {
            int rowShift = rowBase + w;
            int nbrMemBase = rowShift * mapCols;
            int ctrMemBase = rowShift * channel;

            for (int ch = 0; ch < channel; ch++)
            {
                int nbrMemShift = nbrMemBase + ch * wSize;
                int ctrMemShift = ctrMemBase + ch;

#define __get_index(wShift, hShift) \
    cnn_text_get_index(wShift, hShift, w, h, ch, width, height, channel)

                ctrMap[ctrMemShift] = __get_index(0, 0);

                nbrMap[nbrMemShift++] = __get_index(-1, -1);
                nbrMap[nbrMemShift++] = __get_index(0, -1);
                nbrMap[nbrMemShift++] = __get_index(+1, -1);

                nbrMap[nbrMemShift++] = __get_index(-1, 0);
                nbrMap[nbrMemShift++] = __get_index(+1, 0);

                nbrMap[nbrMemShift++] = __get_index(-1, +1);
                nbrMap[nbrMemShift++] = __get_index(0, +1);
                nbrMap[nbrMemShift++] = __get_index(+1, +1);
            }
        }
    }
}

#endif
