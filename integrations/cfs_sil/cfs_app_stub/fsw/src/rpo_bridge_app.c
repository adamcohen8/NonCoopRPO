#include "rpo_bridge_app.h"

RPO_BRIDGE_AppData_t RPO_BRIDGE_AppData;

void RPO_BRIDGE_AppMain(void)
{
    int32 status;
    const CFE_SB_Buffer_t *SBBufPtr;

    RPO_BRIDGE_AppData.RunStatus = CFE_ES_RunStatus_APP_RUN;
    status                       = RPO_BRIDGE_AppInit();

    if (status != CFE_SUCCESS)
    {
        RPO_BRIDGE_AppData.RunStatus = CFE_ES_RunStatus_APP_ERROR;
    }

    while (CFE_ES_RunLoop(&RPO_BRIDGE_AppData.RunStatus))
    {
        status = CFE_SB_ReceiveBuffer(&SBBufPtr, RPO_BRIDGE_AppData.SensorPipe, CFE_SB_PEND_FOREVER);
        if (status == CFE_SUCCESS)
        {
            RPO_BRIDGE_ProcessSensorPacket(SBBufPtr);
        }
    }
}

int32 RPO_BRIDGE_AppInit(void)
{
    int32 status;

    CFE_ES_RegisterApp();
    CFE_PSP_MemSet(&RPO_BRIDGE_AppData, 0, sizeof(RPO_BRIDGE_AppData));

    status = CFE_SB_CreatePipe(&RPO_BRIDGE_AppData.SensorPipe, RPO_BRIDGE_PIPE_DEPTH, RPO_BRIDGE_PIPE_NAME);
    if (status != CFE_SUCCESS)
    {
        return status;
    }

    status = CFE_SB_Subscribe(RPO_BRIDGE_SIM_SENSOR_MID, RPO_BRIDGE_AppData.SensorPipe);
    if (status != CFE_SUCCESS)
    {
        return status;
    }

    return CFE_SUCCESS;
}

void RPO_BRIDGE_ProcessSensorPacket(const CFE_SB_Buffer_t *SBBufPtr)
{
    const RPO_BRIDGE_SimSensor_t *sensor = (const RPO_BRIDGE_SimSensor_t *)SBBufPtr;
    RPO_BRIDGE_ActCmd_t *cmd             = &RPO_BRIDGE_AppData.LastActCmd;

    /*
     * TODO:
     * 1) Replace this hold command with calls into your cFS GNC apps.
     * 2) Populate cmd fields from FSW output, then publish command MID.
     */
    CFE_PSP_MemSet(cmd, 0, sizeof(*cmd));
    cmd->Mode = 1;
    cmd->ValidTimeoutMs = 200;

    CFE_MSG_Init(CFE_MSG_PTR(cmd), RPO_BRIDGE_ACT_CMD_MID, sizeof(RPO_BRIDGE_ActCmd_t));
    CFE_SB_TransmitMsg(CFE_MSG_PTR(cmd), true);
    RPO_BRIDGE_AppData.CmdCounter++;

    (void)sensor;
}

void RPO_BRIDGE_ProcessCommandPacket(const CFE_SB_Buffer_t *SBBufPtr)
{
    (void)SBBufPtr;
}
