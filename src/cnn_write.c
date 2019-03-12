#include <assert.h>

#include "cnn_builtin_math.h"
#include "cnn_strdef.h"
#include "cnn_write.h"

int cnn_export(cnn_t cnn, const char* fPath)
{
    return cnn_export_root(&cnn->cfg, cnn->layerList, fPath);
}

int cnn_config_export(cnn_config_t cfg, const char* fPath)
{
    return cnn_export_root(cfg, NULL, fPath);
}

int cnn_write_config_xml(struct CNN_CONFIG* cfgRef, xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

    // Start config node
    cnn_xml_run(xmlTextWriterStartElement(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_CONFIG]),
                ret, RET);

    // Start input node
    cnn_xml_run(xmlTextWriterStartElement(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_INPUT]),
                ret, RET);

    // Write width
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->width);
    cnn_xml_run(
        xmlTextWriterWriteElement(writer, (xmlChar*)cnn_str_list[CNN_STR_WIDTH],
                                  (xmlChar*)buf),
        ret, RET);

    // Write height
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->height);
    cnn_xml_run(
        xmlTextWriterWriteElement(
            writer, (xmlChar*)cnn_str_list[CNN_STR_HEIGHT], (xmlChar*)buf),
        ret, RET);

    // Write channel
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->channel);
    cnn_xml_run(
        xmlTextWriterWriteElement(
            writer, (xmlChar*)cnn_str_list[CNN_STR_CHANNEL], (xmlChar*)buf),
        ret, RET);

    // End input node
    cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

    // Write batch
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->batch);
    cnn_xml_run(
        xmlTextWriterWriteElement(writer, (xmlChar*)cnn_str_list[CNN_STR_BATCH],
                                  (xmlChar*)buf),
        ret, RET);

    // End config node
    cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

RET:
    return ret;
}

int cnn_write_layer_input_xml(xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;

    // Write layer type
    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
                    (xmlChar*)cnn_str_list[CNN_STR_INPUT]),
                ret, RET);

RET:
    return ret;
}

int cnn_write_layer_fc_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
                           int layerIndex, xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

    // Write layer type
    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
                    (xmlChar*)cnn_str_list[CNN_STR_FC]),
                ret, RET);

    // Write size
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].fc.size);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_SIZE], (xmlChar*)buf),
        ret, RET);

    // Write network detail
    if (layerRef != NULL)
    {
        // Write weight
        cnn_run(cnn_write_mat_xml(&layerRef[layerIndex].fc.weight,
                                  cnn_str_list[CNN_STR_WEIGHT], writer),
                ret, RET);

        // Write bias
        cnn_run(cnn_write_mat_xml(&layerRef[layerIndex].fc.bias,
                                  cnn_str_list[CNN_STR_BIAS], writer),
                ret, RET);
    }

RET:
    return ret;
}

int cnn_write_layer_activ_xml(struct CNN_CONFIG* cfgRef, int layerIndex,
                              xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;

    // Write layer type
    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
                    (xmlChar*)cnn_str_list[CNN_STR_ACTIV]),
                ret, RET);

    // Write activation function id
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_ID],
            (xmlChar*)cnn_activ_name[cfgRef->layerCfg[layerIndex].activ.id]),
        ret, RET);

RET:
    return ret;
}

int cnn_write_layer_conv_xml(struct CNN_CONFIG* cfgRef,
                             union CNN_LAYER* layerRef, int layerIndex,
                             xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

    // Write layer type
    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
                    (xmlChar*)cnn_str_list[CNN_STR_CONV]),
                ret, RET);

    // Write padding
    cnn_run(
        cnn_write_pad_attr_xml(cfgRef->layerCfg[layerIndex].conv.pad, writer),
        ret, RET);

    // Write dimension
    cnn_run(
        cnn_write_dim_attr_xml(cfgRef->layerCfg[layerIndex].conv.dim, writer),
        ret, RET);

    // Write filter
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].conv.filter);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_FILTER], (xmlChar*)buf),
        ret, RET);

    // Write size
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].conv.size);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_SIZE], (xmlChar*)buf),
        ret, RET);

    // Write network detail
    if (layerRef != NULL)
    {
        // Write kernel
        cnn_run(cnn_write_mat_xml(&layerRef[layerIndex].conv.kernel,
                                  cnn_str_list[CNN_STR_KERNEL], writer),
                ret, RET);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
        // Write bias
        cnn_run(cnn_write_mat_xml(&layerRef[layerIndex].conv.bias,
                                  cnn_str_list[CNN_STR_BIAS], writer),
                ret, RET);
#endif
    }

RET:
    return ret;
}

int cnn_write_mat_xml(struct CNN_MAT* matPtr, const char* nodeName,
                      xmlTextWriterPtr writer)
{
    int i;
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

#ifdef CNN_WITH_CUDA
    float tmpVal;
#endif

    // Start node
    cnn_xml_run(xmlTextWriterStartElement(writer, (xmlChar*)nodeName), ret,
                RET);

    for (i = 0; i < matPtr->rows * matPtr->cols; i++)
    {
        // Start value node
        cnn_xml_run(xmlTextWriterStartElement(
                        writer, (xmlChar*)cnn_str_list[CNN_STR_VALUE]),
                    ret, RET);

        // Write index
        cnn_itostr(buf, CNN_XML_BUFLEN, i);
        cnn_xml_run(
            xmlTextWriterWriteAttribute(
                writer, (xmlChar*)cnn_str_list[CNN_STR_INDEX], (xmlChar*)buf),
            ret, RET);

        // Write value
#ifdef CNN_WITH_CUDA
        cnn_run_cu(cudaMemcpy(&tmpVal, matPtr->mat + i, sizeof(float),
                              cudaMemcpyDeviceToHost),
                   ret, RET);
        cnn_ftostr(buf, CNN_XML_BUFLEN, tmpVal);
#else
        cnn_ftostr(buf, CNN_XML_BUFLEN, matPtr->mat[i]);
#endif

        cnn_xml_run(xmlTextWriterWriteString(writer, (xmlChar*)buf), ret, RET);

        // End value node
        cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);
    }

    // End node
    cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

RET:
    return ret;
}

int cnn_write_pad_attr_xml(int pad, xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    const char* str = NULL;

    // Write padding attribute
    switch (pad)
    {
        case CNN_PAD_VALID:
            str = cnn_str_list[CNN_STR_VALID];
            break;

        case CNN_PAD_SAME:
            str = cnn_str_list[CNN_STR_SAME];
            break;

        default:
            assert(!"Invalid padding type");
    }

    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_PAD], (xmlChar*)str),
                ret, RET);

RET:
    return ret;
}

int cnn_write_dim_attr_xml(int dim, xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    const char* str = NULL;

    // Write dimension attribute
    switch (dim)
    {
        case CNN_DIM_1D:
            str = cnn_str_list[CNN_STR_1D];
            break;

        case CNN_DIM_2D:
            str = cnn_str_list[CNN_STR_2D];
            break;

        default:
            assert(!"Invalid dimension type");
    }

    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_DIM], (xmlChar*)str),
                ret, RET);

RET:
    return ret;
}

int cnn_write_layer_pool_xml(struct CNN_CONFIG* cfgRef, int layerIndex,
                             xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

    // Write layer type
    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
                    (xmlChar*)cnn_str_list[CNN_STR_POOL]),
                ret, RET);

    // Write pooling type
    switch (cfgRef->layerCfg[layerIndex].pool.poolType)
    {
        case CNN_POOL_MAX:
            cnn_xml_run(xmlTextWriterWriteAttribute(
                            writer, (xmlChar*)cnn_str_list[CNN_STR_POOL_TYPE],
                            (xmlChar*)cnn_str_list[CNN_STR_MAX]),
                        ret, RET);
            break;

        case CNN_POOL_AVG:
            cnn_xml_run(xmlTextWriterWriteAttribute(
                            writer, (xmlChar*)cnn_str_list[CNN_STR_POOL_TYPE],
                            (xmlChar*)cnn_str_list[CNN_STR_AVG]),
                        ret, RET);
            break;

        default:
            assert(!"Invalid pooling type");
    }

    // Write dimension
    cnn_run(
        cnn_write_dim_attr_xml(cfgRef->layerCfg[layerIndex].pool.dim, writer),
        ret, RET);

    // Write size
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].pool.size);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_SIZE], (xmlChar*)buf),
        ret, RET);

RET:
    return ret;
}

int cnn_write_layer_drop_xml(struct CNN_CONFIG* cfgRef, int layerIndex,
                             xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

    // Write layer type
    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
                    (xmlChar*)cnn_str_list[CNN_STR_DROP]),
                ret, RET);

    // Write size
    cnn_ftostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].drop.rate);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_RATE], (xmlChar*)buf),
        ret, RET);

RET:
    return ret;
}

int cnn_write_layer_bn_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
                           int layerIndex, xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

    // Write layer type
    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
                    (xmlChar*)cnn_str_list[CNN_STR_BN]),
                ret, RET);

    // Write gamma
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].bn.rInit);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_GAMMA], (xmlChar*)buf),
        ret, RET);

    // Write beta
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].bn.bInit);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_BETA], (xmlChar*)buf),
        ret, RET);

    // Write network detail
    if (layerRef != NULL)
    {
        // Write batch normalization parameter
        cnn_run(cnn_write_mat_xml(&layerRef[layerIndex].bn.bnVar,
                                  cnn_str_list[CNN_STR_PARAM], writer),
                ret, RET);
    }

RET:
    return ret;
}

int cnn_write_layer_text_xml(struct CNN_CONFIG* cfgRef,
                             union CNN_LAYER* layerRef, int layerIndex,
                             xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

    // Write layer type
    cnn_xml_run(xmlTextWriterWriteAttribute(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
                    (xmlChar*)cnn_str_list[CNN_STR_TEXT]),
                ret, RET);

    // Write activation function id
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_ID],
            (xmlChar*)
                cnn_activ_name[cfgRef->layerCfg[layerIndex].text.activId]),
        ret, RET);

    // Write filter
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].text.filter);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_FILTER], (xmlChar*)buf),
        ret, RET);

    // Write alpha
    cnn_ftostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].text.aInit);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_ALPHA], (xmlChar*)buf),
        ret, RET);

    // Write network detail
    if (layerRef != NULL)
    {
        // Write weight
        cnn_run(cnn_write_mat_xml(&layerRef[layerIndex].text.weight,
                                  cnn_str_list[CNN_STR_WEIGHT], writer),
                ret, RET);

        // Write bias
        cnn_run(cnn_write_mat_xml(&layerRef[layerIndex].text.bias,
                                  cnn_str_list[CNN_STR_BIAS], writer),
                ret, RET);

        // Write alpha
        cnn_run(cnn_write_mat_xml(&layerRef[layerIndex].text.alpha,
                                  cnn_str_list[CNN_STR_ALPHA], writer),
                ret, RET);
    }

RET:
    return ret;
}

int cnn_write_network_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
                          xmlTextWriterPtr writer)
{
    int ret = CNN_NO_ERROR;
    char buf[CNN_XML_BUFLEN] = {0};

    // Start network node
    cnn_xml_run(xmlTextWriterStartElement(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_NETWORK]),
                ret, RET);

    // Write size (layers)
    cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layers);
    cnn_xml_run(
        xmlTextWriterWriteAttribute(
            writer, (xmlChar*)cnn_str_list[CNN_STR_SIZE], (xmlChar*)buf),
        ret, RET);

    // Write network architecture
    for (int i = 0; i < cfgRef->layers; i++)
    {
        // Start layer node
        cnn_xml_run(xmlTextWriterStartElement(
                        writer, (xmlChar*)cnn_str_list[CNN_STR_LAYER]),
                    ret, RET);

        // Write layer index
        cnn_itostr(buf, CNN_XML_BUFLEN, i);
        cnn_xml_run(
            xmlTextWriterWriteAttribute(
                writer, (xmlChar*)cnn_str_list[CNN_STR_INDEX], (xmlChar*)buf),
            ret, RET);

        // Write layer attribute
        switch (cfgRef->layerCfg[i].type)
        {
            case CNN_LAYER_INPUT:
                cnn_run(cnn_write_layer_input_xml(writer), ret, RET);
                break;

            case CNN_LAYER_FC:
                cnn_run(cnn_write_layer_fc_xml(cfgRef, layerRef, i, writer),
                        ret, RET);
                break;

            case CNN_LAYER_ACTIV:
                cnn_run(cnn_write_layer_activ_xml(cfgRef, i, writer), ret, RET);
                break;

            case CNN_LAYER_CONV:
                cnn_run(cnn_write_layer_conv_xml(cfgRef, layerRef, i, writer),
                        ret, RET);
                break;

            case CNN_LAYER_POOL:
                cnn_run(cnn_write_layer_pool_xml(cfgRef, i, writer), ret, RET);
                break;

            case CNN_LAYER_DROP:
                cnn_run(cnn_write_layer_drop_xml(cfgRef, i, writer), ret, RET);
                break;

            case CNN_LAYER_BN:
                cnn_run(cnn_write_layer_bn_xml(cfgRef, layerRef, i, writer),
                        ret, RET);
                break;

            case CNN_LAYER_TEXT:
                cnn_run(cnn_write_layer_text_xml(cfgRef, layerRef, i, writer),
                        ret, RET);
                break;

            default:
                assert(!"Invalid layer type");
        }

        // End layer node
        cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);
    }

    // End network node
    cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

RET:
    return ret;
}

int cnn_export_root(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
                    const char* fPath)
{
    int ret = CNN_NO_ERROR;

    xmlTextWriterPtr writer = NULL;

    // Create xml writer
    writer = xmlNewTextWriterFilename(fPath, 0);
    if (writer == NULL)
    {
        ret = CNN_FILE_OP_FAILED;
        goto RET;
    }

    cnn_xml_run(xmlTextWriterSetIndent(writer, 1), ret, RET);
    cnn_xml_run(xmlTextWriterStartDocument(writer, CNN_XML_VER_STR,
                                           CNN_XML_ENC_STR, NULL),
                ret, RET);

    // Write root
    cnn_xml_run(xmlTextWriterStartElement(
                    writer, (xmlChar*)cnn_str_list[CNN_STR_MODEL]),
                ret, RET);

    // Write config
    cnn_run(cnn_write_config_xml(cfgRef, writer), ret, RET);

    // Write netowrk
    cnn_run(cnn_write_network_xml(cfgRef, layerRef, writer), ret, RET);

    // End root
    cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);
    cnn_xml_run(xmlTextWriterEndDocument(writer), ret, RET);

RET:
    if (writer != NULL)
    {
        xmlFreeTextWriter(writer);
    }

    return ret;
}
