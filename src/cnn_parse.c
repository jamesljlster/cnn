#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cnn_builtin_math.h"
#include "cnn_init.h"
#include "cnn_parse.h"
#include "cnn_strdef.h"

void cnn_ftostr(char* buf, int bufSize, float val)
{
    int __ret = snprintf(buf, bufSize, "%.32g", val);
    assert(__ret > 0 && __ret <= bufSize && "Insufficient buffer size");
}

void cnn_itostr(char* buf, int bufSize, int val)
{
    int __ret = snprintf(buf, bufSize, "%d", val);
    assert(__ret > 0 && __ret <= bufSize && "Insufficient buffer size");
}

int cnn_config_import(cnn_config_t* cfgPtr, const char* fPath)
{
    int ret = CNN_NO_ERROR;
    cnn_config_t tmpCfg = NULL;

    // Create config
    cnn_run(cnn_config_create(&tmpCfg), ret, RET);

    // Import config
    cnn_run(cnn_import_root(tmpCfg, NULL, fPath), ret, ERR);

    // Assign value
    *cfgPtr = tmpCfg;

    goto RET;

ERR:
    cnn_config_delete(tmpCfg);

RET:
    return ret;
}

int cnn_import(cnn_t* cnnPtr, const char* fPath)
{
    int ret = CNN_NO_ERROR;
    cnn_t tmpCnn = NULL;

    // Memory allocation
    cnn_alloc(tmpCnn, 1, struct CNN, ret, RET);

    // Import
    cnn_run(cnn_import_root(&tmpCnn->cfg, &tmpCnn->layerList, fPath), ret, ERR);

    // Assign value
    *cnnPtr = tmpCnn;

    goto RET;

ERR:
    cnn_delete(tmpCnn);

RET:
    return ret;
}

int cnn_parse_network_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
    int ret = CNN_NO_ERROR;
    int strId;
    int tmp;

    xmlNodePtr cur;
    xmlAttrPtr attrCur, layers;

    xmlChar* xStr = NULL;

    // Find layers
    layers = NULL;
    attrCur = node->properties;
    while (attrCur != NULL)
    {
        // Find attr id
        strId = cnn_strdef_get_id((const char*)attrCur->name);
        switch (strId)
        {
            case CNN_STR_SIZE:
                layers = attrCur;
                break;
        }

        attrCur = attrCur->next;
    }

    if (layers == NULL)
    {
        ret = CNN_INFO_NOT_FOUND;
        goto RET;
    }

    // Parse and set layers
    xStr = xmlNodeGetContent(layers->children);
    cnn_run(cnn_strtoi(&tmp, (const char*)xStr), ret, RET);
    xmlFree(xStr);
    xStr = NULL;

    cnn_run(cnn_config_set_layers(cfgPtr, tmp), ret, RET);

    cur = node->xmlChildrenNode;
    while (cur != NULL)
    {
        // Find node id
        strId = cnn_strdef_get_id((const char*)cur->name);
        switch (strId)
        {
            case CNN_STR_LAYER:
                cnn_run(cnn_parse_network_layer_xml(cfgPtr, cur), ret, RET);
                break;
        }

        cur = cur->next;
    }

RET:
    xmlFree(xStr);
    return ret;
}

int cnn_parse_network_layer_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
    int ret = CNN_NO_ERROR;
    int tmpType = -1, tmpIndex;
    int strId;

    xmlAttrPtr attrCur;
    xmlAttrPtr index = NULL, type = NULL;
    xmlAttrPtr pad = NULL, dim = NULL, size = NULL, poolType = NULL, id = NULL,
               rate = NULL, filter = NULL, gamma = NULL, beta = NULL,
               expAvgF = NULL;

    xmlChar* xStr = NULL;

    // Parse attribute
    attrCur = node->properties;
    while (attrCur != NULL)
    {
        // Find node id
        strId = cnn_strdef_get_id((const char*)attrCur->name);
        switch (strId)
        {
            case CNN_STR_INDEX:
                index = attrCur;
                break;

            case CNN_STR_TYPE:
                type = attrCur;
                break;

            case CNN_STR_DIM:
                dim = attrCur;
                break;

            case CNN_STR_POOL_TYPE:
                poolType = attrCur;
                break;

            case CNN_STR_ID:
                id = attrCur;
                break;

            case CNN_STR_SIZE:
                size = attrCur;
                break;

            case CNN_STR_RATE:
                rate = attrCur;
                break;

            case CNN_STR_FILTER:
                filter = attrCur;
                break;

            case CNN_STR_PAD:
                pad = attrCur;
                break;

            case CNN_STR_GAMMA:
                gamma = attrCur;
                break;

            case CNN_STR_BETA:
                beta = attrCur;
                break;

            case CNN_STR_EAF:
                expAvgF = attrCur;
                break;
        }

        attrCur = attrCur->next;
    }

    if (index == NULL || type == NULL)
    {
        ret = CNN_INFO_NOT_FOUND;
        goto RET;
    }

    // Parse type and index
    xStr = xmlNodeGetContent(index->children);
    cnn_run(cnn_strtoi(&tmpIndex, (const char*)xStr), ret, RET);
    xmlFree(xStr);

    xStr = xmlNodeGetContent(type->children);
    strId = cnn_strdef_get_id((const char*)xStr);
    xmlFree(xStr);
    xStr = NULL;

    switch (strId)
    {
        case CNN_STR_INPUT:
            tmpType = CNN_LAYER_INPUT;
            break;

        case CNN_STR_FC:
            tmpType = CNN_LAYER_FC;
            break;

        case CNN_STR_ACTIV:
            tmpType = CNN_LAYER_ACTIV;
            break;

        case CNN_STR_CONV:
            tmpType = CNN_LAYER_CONV;
            break;

        case CNN_STR_POOL:
            tmpType = CNN_LAYER_POOL;
            break;

        case CNN_STR_DROP:
            tmpType = CNN_LAYER_DROP;
            break;

        case CNN_STR_BN:
            tmpType = CNN_LAYER_BN;
            break;

        default:
            assert(!"Invalid layer type");
    }

    // Parse layer
    cfgPtr->layerCfg[tmpIndex].type = tmpType;
    switch (tmpType)
    {
        case CNN_LAYER_INPUT:
            break;

        case CNN_LAYER_FC:
            if (size == NULL)
            {
                ret = CNN_INFO_NOT_FOUND;
                goto RET;
            }

            // Parse size
            xStr = xmlNodeGetContent(size->children);
            cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].fc.size,
                               (const char*)xStr),
                    ret, RET);
            xmlFree(xStr);
            xStr = NULL;
            break;

        case CNN_LAYER_ACTIV:
            if (id == NULL)
            {
                ret = CNN_INFO_NOT_FOUND;
                goto RET;
            }

            // Parse id
            xStr = xmlNodeGetContent(id->children);
            strId = cnn_get_activ_id((const char*)xStr);
            xmlFree(xStr);
            xStr = NULL;

            assert(strId >= 0 && "Invalid activation function ID");
            cfgPtr->layerCfg[tmpIndex].activ.id = strId;

            break;

        case CNN_LAYER_CONV:
            if (pad == NULL || dim == NULL || size == NULL || filter == NULL)
            {
                ret = CNN_INFO_NOT_FOUND;
                goto RET;
            }

            // Parse padding
            xStr = xmlNodeGetContent(pad->children);
            strId = cnn_strdef_get_id((const char*)xStr);
            xmlFree(xStr);
            xStr = NULL;

            switch (strId)
            {
                case CNN_STR_VALID:
                    cfgPtr->layerCfg[tmpIndex].conv.pad = CNN_PAD_VALID;
                    break;

                case CNN_STR_SAME:
                    cfgPtr->layerCfg[tmpIndex].conv.pad = CNN_PAD_SAME;
                    break;

                default:
                    assert(!"Invalid padding type");
            }

            // Parse dimension
            xStr = xmlNodeGetContent(dim->children);
            strId = cnn_strdef_get_id((const char*)xStr);
            xmlFree(xStr);
            xStr = NULL;

            switch (strId)
            {
                case CNN_STR_1D:
                    cfgPtr->layerCfg[tmpIndex].conv.dim = CNN_DIM_1D;
                    break;

                case CNN_STR_2D:
                    cfgPtr->layerCfg[tmpIndex].conv.dim = CNN_DIM_2D;
                    break;

                default:
                    assert(!"Invalid dimension type");
            }

            // Parse filter
            xStr = xmlNodeGetContent(filter->children);
            cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].conv.filter,
                               (const char*)xStr),
                    ret, RET);
            xmlFree(xStr);

            // Parse size
            xStr = xmlNodeGetContent(size->children);
            cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].conv.size,
                               (const char*)xStr),
                    ret, RET);
            xmlFree(xStr);
            xStr = NULL;
            break;

        case CNN_LAYER_POOL:
            if (poolType == NULL || dim == NULL || size == NULL)
            {
                ret = CNN_INFO_NOT_FOUND;
                goto RET;
            }

            // Parse pooling type
            xStr = xmlNodeGetContent(poolType->children);
            strId = cnn_strdef_get_id((const char*)xStr);
            xmlFree(xStr);
            xStr = NULL;

            switch (strId)
            {
                case CNN_STR_MAX:
                    cfgPtr->layerCfg[tmpIndex].pool.poolType = CNN_POOL_MAX;
                    break;

                case CNN_STR_AVG:
                    cfgPtr->layerCfg[tmpIndex].pool.poolType = CNN_POOL_AVG;
                    break;

                default:
                    assert(!"Invalid pooling type");
            }

            // Parse dimension
            xStr = xmlNodeGetContent(dim->children);
            strId = cnn_strdef_get_id((const char*)xStr);
            xmlFree(xStr);
            xStr = NULL;
            switch (strId)
            {
                case CNN_STR_1D:
                    cfgPtr->layerCfg[tmpIndex].pool.dim = CNN_DIM_1D;
                    break;

                case CNN_STR_2D:
                    cfgPtr->layerCfg[tmpIndex].pool.dim = CNN_DIM_2D;
                    break;

                default:
                    assert(!"Invalid dimension type");
            }

            // Parse size
            xStr = xmlNodeGetContent(size->children);
            cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].pool.size,
                               (const char*)xStr),
                    ret, RET);
            xmlFree(xStr);
            xStr = NULL;

            break;

        case CNN_LAYER_DROP:
            if (rate == NULL)
            {
                ret = CNN_INFO_NOT_FOUND;
                goto RET;
            }

            // Parse rate
            xStr = xmlNodeGetContent(rate->children);
            cnn_run(cnn_strtof(&cfgPtr->layerCfg[tmpIndex].drop.rate,
                               (const char*)xStr),
                    ret, RET);
            xmlFree(xStr);
            xStr = NULL;

            // Set dropout scale
            cfgPtr->layerCfg[tmpIndex].drop.scale =
                1.0 / (1.0 - cfgPtr->layerCfg[tmpIndex].drop.rate);

            break;

        case CNN_LAYER_BN:
            if (gamma == NULL || beta == NULL || expAvgF == NULL)
            {
                ret = CNN_INFO_NOT_FOUND;
                goto RET;
            }

            // Parse gamma
            xStr = xmlNodeGetContent(gamma->children);
            cnn_run(cnn_strtof(&cfgPtr->layerCfg[tmpIndex].bn.rInit,
                               (const char*)xStr),
                    ret, RET);
            xmlFree(xStr);
            xStr = NULL;

            // Parse beta
            xStr = xmlNodeGetContent(beta->children);
            cnn_run(cnn_strtof(&cfgPtr->layerCfg[tmpIndex].bn.bInit,
                               (const char*)xStr),
                    ret, RET);
            xmlFree(xStr);
            xStr = NULL;

            // Parse exponential average factor
            xStr = xmlNodeGetContent(expAvgF->children);
            cnn_run(cnn_strtof(&cfgPtr->layerCfg[tmpIndex].bn.expAvgFactor,
                               (const char*)xStr),
                    ret, RET);
            xmlFree(xStr);
            xStr = NULL;

            break;

        default:
            assert(!"Invalid layer type");
    }

RET:
    xmlFree(xStr);
    return ret;
}

int cnn_parse_config_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
    int ret = CNN_NO_ERROR;
    int strId;

    xmlNodePtr cur;
    xmlNodePtr input;  // Must exist
    xmlNodePtr batch;

    xmlChar* xStr = NULL;

    // Find node
    input = NULL;
    batch = NULL;
    cur = node->xmlChildrenNode;
    while (cur != NULL)
    {
        // Find node id
        strId = cnn_strdef_get_id((const char*)cur->name);
        switch (strId)
        {
            case CNN_STR_INPUT:
                input = cur;
                break;

            case CNN_STR_BATCH:
                batch = cur;
                break;
        }

        cur = cur->next;
    }

    if (input == NULL)
    {
        ret = CNN_INFO_NOT_FOUND;
        goto RET;
    }

    // Parse input
    cnn_run(cnn_parse_config_input_xml(cfgPtr, input), ret, RET);

    // Parse batch
    if (batch != NULL)
    {
        xStr = xmlNodeGetContent(batch);
        cnn_run(cnn_strtoi(&cfgPtr->batch, (const char*)xStr), ret, RET);
        xmlFree(xStr);
        xStr = NULL;
    }

RET:
    xmlFree(xStr);
    return ret;
}

int cnn_parse_config_input_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
    int ret = CNN_NO_ERROR;
    int strId;

    xmlNodePtr cur;
    xmlNodePtr width, height, channel;

    xmlChar* xStr = NULL;

    // Find node
    width = NULL;
    height = NULL;
    channel = NULL;
    cur = node->xmlChildrenNode;
    while (cur != NULL)
    {
        // Find node id
        strId = cnn_strdef_get_id((const char*)cur->name);
        switch (strId)
        {
            case CNN_STR_WIDTH:
                width = cur;
                break;

            case CNN_STR_HEIGHT:
                height = cur;
                break;

            case CNN_STR_CHANNEL:
                channel = cur;
                break;
        }

        cur = cur->next;
    }

    if (width == NULL || height == NULL || channel == NULL)
    {
        ret = CNN_INFO_NOT_FOUND;
        goto RET;
    }

    // Parse information
    xStr = xmlNodeGetContent(width);
    cnn_run(cnn_strtoi(&cfgPtr->width, (const char*)xStr), ret, RET);
    xmlFree(xStr);

    xStr = xmlNodeGetContent(height);
    cnn_run(cnn_strtoi(&cfgPtr->height, (const char*)xStr), ret, RET);
    xmlFree(xStr);

    xStr = xmlNodeGetContent(channel);
    cnn_run(cnn_strtoi(&cfgPtr->channel, (const char*)xStr), ret, RET);

RET:
    xmlFree(xStr);
    return ret;
}

int cnn_import_root(struct CNN_CONFIG* cfgPtr, union CNN_LAYER** layerPtr,
                    const char* fPath)
{
    int ret = CNN_NO_ERROR;
    int strId;

    struct CNN tmpCnn;

    xmlDocPtr doc = NULL;
    xmlNodePtr cur, cfg, net;

    // Zero memory
    memset(&tmpCnn, 0, sizeof(struct CNN));

    // Parse xml
    doc = xmlParseFile(fPath);
    if (doc == NULL)
    {
        ret = CNN_FILE_OP_FAILED;
        goto RET;
    }

    // Check xml file
    cur = xmlDocGetRootElement(doc);
    if (cur == NULL)
    {
        ret = CNN_INVALID_FILE;
        goto RET;
    }

    if (xmlStrcmp(cur->name, (xmlChar*)cnn_str_list[CNN_STR_MODEL]))
    {
        ret = CNN_INVALID_FILE;
        goto RET;
    }

    // Find config and network node
    cfg = NULL;
    net = NULL;
    cur = cur->xmlChildrenNode;
    while (cur != NULL)
    {
        // Find node id
        strId = cnn_strdef_get_id((const char*)cur->name);
        switch (strId)
        {
            case CNN_STR_CONFIG:
                cfg = cur;
                break;

            case CNN_STR_NETWORK:
                net = cur;
                break;
        }

        cur = cur->next;
    }

    if (cfg == NULL || net == NULL)
    {
        ret = CNN_INFO_NOT_FOUND;
        goto RET;
    }

    // Parse config
    cnn_run(cnn_parse_config_xml(cfgPtr, cfg), ret, RET);

    // Parse network config
    cnn_run(cnn_parse_network_xml(cfgPtr, net), ret, RET);

    // Parse network detail
    if (layerPtr != NULL)
    {
        // Allocate network
        cnn_run(cnn_config_struct_clone(&tmpCnn.cfg, cfgPtr), ret, RET);
        cnn_run(cnn_network_alloc(&tmpCnn), ret, RET);

        // Parsing
        cnn_run(cnn_parse_network_detail_xml(&tmpCnn, doc), ret, RET);

#ifdef CNN_WITH_CUDA
        // Allocate workspace size
        cnn_run(cnn_cudnn_ws_alloc(), ret, RET);
#endif

        // Assign value
        *layerPtr = tmpCnn.layerList;
        tmpCnn.layerList = NULL;
    }

RET:
    if (doc != NULL)
    {
        xmlFreeDoc(doc);
    }

    cnn_struct_delete(&tmpCnn);

    return ret;
}

int cnn_parse_network_detail_xml(struct CNN* cnn, xmlDocPtr doc)
{
    int i;
    int ret = CNN_NO_ERROR;

    char buf[CNN_XML_BUFLEN] = {0};

    xmlXPathContextPtr cont;
    xmlXPathObjectPtr obj = NULL;

    // Create XPath context
    cont = xmlXPathNewContext(doc);
    if (cont == NULL)
    {
        ret = CNN_MEM_FAILED;
        goto RET;
    }

    // Parse layer detail
    for (i = 0; i < cnn->cfg.layers; i++)
    {
        switch (cnn->cfg.layerCfg[i].type)
        {
            case CNN_LAYER_FC:
                // Select node
                ret = snprintf(buf, CNN_XML_BUFLEN, "/%s/%s/%s[@%s='%d']",
                               cnn_str_list[CNN_STR_MODEL],
                               cnn_str_list[CNN_STR_NETWORK],
                               cnn_str_list[CNN_STR_LAYER],
                               cnn_str_list[CNN_STR_INDEX], i);
                assert(ret > 0 && ret < CNN_XML_BUFLEN &&
                       "Insufficient buffer size");

                obj = xmlXPathEval((xmlChar*)buf, cont);
                if (obj == NULL)
                {
                    ret = CNN_MEM_FAILED;
                    goto RET;
                }

                // Parse
                if (!xmlXPathNodeSetIsEmpty(obj->nodesetval))
                {
                    cnn_run(cnn_parse_network_detail_fc_xml(
                                cnn, i, obj->nodesetval->nodeTab[0]),
                            ret, RET);
                }

                break;

            case CNN_LAYER_CONV:
                // Select node
                ret = snprintf(buf, CNN_XML_BUFLEN, "/%s/%s/%s[@%s='%d']",
                               cnn_str_list[CNN_STR_MODEL],
                               cnn_str_list[CNN_STR_NETWORK],
                               cnn_str_list[CNN_STR_LAYER],
                               cnn_str_list[CNN_STR_INDEX], i);
                assert(ret > 0 && ret < CNN_XML_BUFLEN &&
                       "Insufficient buffer size");

                obj = xmlXPathEval((xmlChar*)buf, cont);
                if (obj == NULL)
                {
                    ret = CNN_MEM_FAILED;
                    goto RET;
                }

                // Parse
                if (!xmlXPathNodeSetIsEmpty(obj->nodesetval))
                {
                    cnn_run(cnn_parse_network_detail_conv_xml(
                                cnn, i, obj->nodesetval->nodeTab[0]),
                            ret, RET);
                }

                break;

            case CNN_LAYER_BN:
                // Select node
                ret = snprintf(buf, CNN_XML_BUFLEN, "/%s/%s/%s[@%s='%d']",
                               cnn_str_list[CNN_STR_MODEL],
                               cnn_str_list[CNN_STR_NETWORK],
                               cnn_str_list[CNN_STR_LAYER],
                               cnn_str_list[CNN_STR_INDEX], i);
                assert(ret > 0 && ret < CNN_XML_BUFLEN &&
                       "Insufficient buffer size");

                obj = xmlXPathEval((xmlChar*)buf, cont);
                if (obj == NULL)
                {
                    ret = CNN_MEM_FAILED;
                    goto RET;
                }

                // Parse
                if (!xmlXPathNodeSetIsEmpty(obj->nodesetval))
                {
                    cnn_run(cnn_parse_network_detail_bn_xml(
                                cnn, i, obj->nodesetval->nodeTab[0]),
                            ret, RET);
                }

                break;

            case CNN_LAYER_INPUT:
            case CNN_LAYER_ACTIV:
            case CNN_LAYER_POOL:
            case CNN_LAYER_DROP:
                break;
        }

        if (obj != NULL)
        {
            xmlXPathFreeObject(obj);
            obj = NULL;
        }
    }

RET:
    if (cont != NULL)
    {
        xmlXPathFreeContext(cont);
    }

    return ret;
}

int cnn_parse_network_detail_fc_xml(struct CNN* cnn, int layerIndex,
                                    xmlNodePtr node)
{
    int ret = CNN_NO_ERROR;
    int strId;

    xmlNodePtr cur;

    // Parsing
    cur = node->xmlChildrenNode;
    while (cur != NULL)
    {
        strId = cnn_strdef_get_id((const char*)cur->name);
        switch (strId)
        {
            case CNN_STR_WEIGHT:
                cnn_run(
                    cnn_parse_mat(&cnn->layerList[layerIndex].fc.weight, cur),
                    ret, RET);
                break;

            case CNN_STR_BIAS:
                cnn_run(cnn_parse_mat(&cnn->layerList[layerIndex].fc.bias, cur),
                        ret, RET);
                break;
        }

        cur = cur->next;
    }

RET:
    return ret;
}

int cnn_parse_network_detail_conv_xml(struct CNN* cnn, int layerIndex,
                                      xmlNodePtr node)
{
    int ret = CNN_NO_ERROR;
    int strId;

    xmlNodePtr cur;

    // Parsing
    cur = node->xmlChildrenNode;
    while (cur != NULL)
    {
        strId = cnn_strdef_get_id((const char*)cur->name);
        switch (strId)
        {
            case CNN_STR_KERNEL:
                cnn_run(
                    cnn_parse_mat(&cnn->layerList[layerIndex].conv.kernel, cur),
                    ret, RET);
                break;

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
            case CNN_STR_BIAS:
                cnn_run(
                    cnn_parse_mat(&cnn->layerList[layerIndex].conv.bias, cur),
                    ret, RET);
                break;
#endif
        }

        cur = cur->next;
    }

RET:
    return ret;
}

int cnn_parse_network_detail_bn_xml(struct CNN* cnn, int layerIndex,
                                    xmlNodePtr node)
{
    int ret = CNN_NO_ERROR;
    int strId;

    xmlNodePtr cur;

    // Parsing
    cur = node->xmlChildrenNode;
    while (cur != NULL)
    {
        strId = cnn_strdef_get_id((const char*)cur->name);
        switch (strId)
        {
            case CNN_STR_GAMMA:
                cnn_run(
                    cnn_parse_mat(&cnn->layerList[layerIndex].bn.bnScale, cur),
                    ret, RET);
                break;

            case CNN_STR_BETA:
                cnn_run(
                    cnn_parse_mat(&cnn->layerList[layerIndex].bn.bnBias, cur),
                    ret, RET);
                break;

            case CNN_STR_MEAN:
                cnn_run(
                    cnn_parse_mat(&cnn->layerList[layerIndex].bn.runMean, cur),
                    ret, RET);
                break;

            case CNN_STR_VAR:
                cnn_run(
                    cnn_parse_mat(&cnn->layerList[layerIndex].bn.runVar, cur),
                    ret, RET);
                break;
        }

        cur = cur->next;
    }

RET:
    return ret;
}

int cnn_parse_mat(struct CNN_MAT* mat, xmlNodePtr node)
{
    int ret = CNN_NO_ERROR;
    int strId;
    int tmpIndex;

#ifdef CNN_WITH_CUDA
    float tmpVal;
#endif

    xmlNodePtr cur;

    xmlChar* xStr = NULL;

    cur = node->xmlChildrenNode;
    while (cur != NULL)
    {
        strId = cnn_strdef_get_id((const char*)cur->name);
        switch (strId)
        {
            case CNN_STR_VALUE:
                // Get value index
                xStr = xmlGetProp(cur, (xmlChar*)cnn_str_list[CNN_STR_INDEX]);
                cnn_run(cnn_strtoi(&tmpIndex, (const char*)xStr), ret, RET);
                xmlFree(xStr);
                xStr = NULL;

                // Parse value
                if (tmpIndex >= 0 && tmpIndex < mat->rows * mat->cols)
                {
                    xStr = xmlNodeGetContent(cur);

#ifdef CNN_WITH_CUDA
                    cnn_run(cnn_strtof(&tmpVal, (const char*)xStr), ret, RET);
                    cnn_run_cu(
                        cudaMemcpy(mat->mat + tmpIndex, &tmpVal, sizeof(float),
                                   cudaMemcpyHostToDevice),
                        ret, RET);
#else
                    cnn_run(cnn_strtof(&mat->mat[tmpIndex], (const char*)xStr),
                            ret, RET);
#endif
                    xmlFree(xStr);
                    xStr = NULL;
                }

                break;
        }

        cur = cur->next;
    }

RET:
    xmlFree(xStr);
    return ret;
}
