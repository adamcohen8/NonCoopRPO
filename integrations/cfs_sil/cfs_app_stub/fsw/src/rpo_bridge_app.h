#ifndef RPO_BRIDGE_APP_H
#define RPO_BRIDGE_APP_H

#include "cfe.h"
#include "rpo_bridge_msg.h"

#define RPO_BRIDGE_APP_NAME "RPO_BRIDGE"
#define RPO_BRIDGE_PIPE_NAME "RPO_BRIDGE_PIPE"
#define RPO_BRIDGE_PIPE_DEPTH 16

typedef struct
{
    CFE_SB_PipeId_t CmdPipe;
    CFE_SB_PipeId_t SensorPipe;
    CFE_ES_RunStatus_t RunStatus;
    uint32 CmdCounter;
    uint32 ErrCounter;
    RPO_BRIDGE_ActCmd_t LastActCmd;
} RPO_BRIDGE_AppData_t;

void RPO_BRIDGE_AppMain(void);
int32 RPO_BRIDGE_AppInit(void);
void RPO_BRIDGE_ProcessCommandPacket(const CFE_SB_Buffer_t *SBBufPtr);
void RPO_BRIDGE_ProcessSensorPacket(const CFE_SB_Buffer_t *SBBufPtr);

extern RPO_BRIDGE_AppData_t RPO_BRIDGE_AppData;

#endif
