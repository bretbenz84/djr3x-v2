"""Visual mirror of head_nano's 8x10 physical-RGB mouth equalizer.

Uses the same audio levels, colors and ballistics. Firmware wobble phases are
local to the Arduino, so this is a command-level mirror, not pixel readback.
"""
import math
import numpy as np
COLORS={'neutral':(255,140,0),'happy':(0,200,255),'excited':(255,200,0),'sad':(40,0,200),'angry':(255,0,0),'curious':(180,0,255)}

class MouthAnimation:
    def __init__(self):
        self.height=np.zeros(8);self.peak=np.zeros(8);self.phase=np.arange(8)*.73
        self.last=None;self.utterance=None;self.last_pixels=np.zeros((80,3));self.fade_pixels=None
        self.last_mode='off'

    def render(self,state,now):
        dt=min(.1,max(0,now-self.last)) if self.last is not None else .033;self.last=now
        mode=state.get('mouth_mode','off');age=max(0,now-state.get('mouth_updated_at',now))
        rgb=np.array(COLORS.get(state.get('mouth_emotion','neutral'),COLORS['neutral']),float)/255
        if mode=='speak' and age>1.5:mode='active'
        if state.get('mouth_utterance')!=self.utterance:
            self.height[:]=0;self.peak[:]=0;self.utterance=state.get('mouth_utterance')
        values=np.zeros((10,8))
        if mode=='speak':
            level=max(0,min(255,state.get('mouth_level',0)))/255
            self.phase+=(2.2+.55*np.arange(8))*(.6+1.4*level)*dt
            target=np.minimum(5,level*(.6+.4*np.sin(self.phase))*np.array([.55,.75,.92,1,1,.92,.75,.55])*5.2)
            self.height+=(target-self.height)*np.minimum(1,np.where(target>self.height,18,6)*dt)
            self.peak=np.where(self.height>self.peak,self.height,np.maximum(0,self.peak-3.5*dt))
            for d in range(5):
                bright=np.clip(self.height-d,0,1)*.45
                if d==0:bright=np.maximum(.1,bright)
                bright=np.where((self.peak>self.height+.6)&(np.minimum(4,self.peak.astype(int))==d),np.maximum(bright,.315),bright)
                values[4-d]=bright;values[5+d]=bright
        elif mode in ('active','idle'):
            values[:]=.075+.025*math.sin(now)
        elif mode in ('sleep','charge'):
            phase=(now%8)/8;values[:]=(.3*(phase*2 if phase<.5 else 2-phase*2))
            soc=state.get('mouth_soc',0)
            rgb=np.array((255,0,0) if mode=='sleep' or soc<=25 else (255,96,0) if soc<=50 else (255,220,0) if soc<=75 else (0,255,0) if soc<=90 else (0,0,255),float)/255
        if mode=='fadeoff':
            if self.last_mode!='fadeoff':self.fade_pixels=self.last_pixels.copy()
            result=(self.fade_pixels if self.fade_pixels is not None else self.last_pixels)*max(0,1-age/4)
        else:result=values.reshape(80,1)*rgb
        self.last_mode=mode;self.last_pixels=result.copy()
        return result
