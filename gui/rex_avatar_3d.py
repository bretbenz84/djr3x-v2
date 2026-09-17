"""Native Qt Quick 3D avatar driven by the existing read-only dashboard snapshot."""
from __future__ import annotations
import json
import math
import time
from functools import lru_cache
import numpy as np
from PySide6.QtCore import QByteArray, QObject, Property, QTimer, QUrl, Signal, Qt
from PySide6.QtGui import QColor, QMatrix3x3, QQuaternion, QVector3D
from PySide6.QtQuick3D import QQuick3DGeometry
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QVBoxLayout
import config
from gui.mouth_leds import MouthAnimation
from gui.avatar_rig import ASSET_DIR, SPEC, pose_matrices, spring_points, point
from gui.rex_avatar import RexAvatar as AvatarState, normalize_servo, chest_render_state, _eye_color, _chest_gauge_color, _prand, _pick

@lru_cache(maxsize=1)
def _geometry_data():
    with np.load(ASSET_DIR / 'geometry.npz', allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}

class DisplayGeometry(QQuick3DGeometry):
    def __init__(self, vertices):
        super().__init__()
        self.setStride(32)
        self.setPrimitiveType(self.PrimitiveType.Triangles)
        self.addAttribute(self.Attribute.Semantic.PositionSemantic, 0, self.Attribute.ComponentType.F32Type)
        self.addAttribute(self.Attribute.Semantic.NormalSemantic, 12, self.Attribute.ComponentType.F32Type)
        self.addAttribute(self.Attribute.Semantic.TexCoordSemantic, 24, self.Attribute.ComponentType.F32Type)
        self.replace(vertices)

    def replace(self, vertices):
        uv = np.column_stack(((vertices[:,0]*.786 + vertices[:,1]*.618 + .15)/.30, np.full(len(vertices), .5)))
        a = np.ascontiguousarray(np.concatenate((vertices, uv), axis=1), dtype='<f4')
        self.setVertexData(QByteArray(a.tobytes()))
        self.setBounds(QVector3D(*a[:, :3].min(axis=0)), QVector3D(*a[:, :3].max(axis=0)))
        self.update()

def tube_vertices(points, radius, sides=8):
    """Round tube following an updated centreline, without stretching wire diameter."""
    p = np.asarray(points)
    tangent = np.gradient(p, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    reference = np.tile([0., 0., 1.], (len(p), 1))
    reference[np.abs(tangent[:, 2]) > .95] = [1., 0., 0.]
    u = np.cross(tangent, reference)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(tangent, u)
    a = np.arange(sides) * 2*math.pi/sides
    normals = u[:, None, :] * np.cos(a)[None, :, None] + v[:, None, :] * np.sin(a)[None, :, None]
    vertices = np.concatenate((p[:, None, :] + radius*normals, normals), axis=2).reshape(-1, 6)
    i = np.arange(len(p)-1)[:, None]*sides + np.arange(sides)[None, :]
    j = np.arange(len(p)-1)[:, None]*sides + (np.arange(sides)[None, :]+1)%sides
    indices = np.stack((i, j, i+sides, j, j+sides, i+sides), axis=-1).reshape(-1)
    return vertices[indices].astype('<f4')

class RigViewState(QObject):
    changed = Signal()
    def __init__(self):
        super().__init__()
        self._transforms = {}
        self._eye = QColor('black')
        self.mouth_animation = MouthAnimation()
        self._mouth = [QColor('black') for _ in range(80)]
        self._led = [QColor('black') for _ in range(9)]
        self._background = True
        self.geometries = []
        self._batches = []
        for batch in json.loads((ASSET_DIR/'materials.json').read_text()):
            geometry = DisplayGeometry(_geometry_data()[batch['id']])
            self.geometries.append(geometry)
            # Blender stores linear RGB; QColor/QML base colors are sRGB.
            srgb = [12.92*c if c <= .0031308 else 1.055*c**(1/2.4)-.055 for c in batch['color']]
            color = QColor.fromRgbF(*srgb)
            self._batches.append(dict(batch, geometry=geometry, color=color))
        self.spring = DisplayGeometry(tube_vertices(spring_points(pose_matrices({})), .005))
        self.geometries.append(self.spring)
        for name, geometry, color in [('spring', self.spring, '#10181c')]:
            self._batches.append(dict(id=name, group='fixed', geometry=geometry, color=QColor(color), metalness=.6, roughness=.35, kind='surface', lamp=-1))

    batches = Property('QVariantList', lambda self: self._batches, constant=True)
    transforms = Property('QVariantMap', lambda self: self._transforms, notify=changed)
    mouthColors = Property('QVariantList', lambda self: self._mouth, notify=changed)
    eyeColor = Property(QColor, lambda self: self._eye, notify=changed)
    ledColors = Property('QVariantList', lambda self: self._led, notify=changed)
    background = Property(bool, lambda self: self._background, notify=changed)

    def pose(self, norms):
        matrices = pose_matrices(norms)
        self._transforms = {
            name: dict(position=QVector3D(*m[:3, 3]), rotation=QQuaternion.fromRotationMatrix(QMatrix3x3(m[:3, :3].reshape(-1).tolist())))
            for name, m in matrices.items()
        }
        self.spring.replace(tube_vertices(spring_points(matrices), .005))

class RexAvatar(AvatarState):
    """Same widget/snapshot contract as the previous avatar, with a 3D scene."""
    def __init__(self, parent=None, *, show_background=True, show_grid=True):
        super().__init__(parent, show_background=show_background, show_grid=show_grid)
        # Gravity-rest elbow, closed visor; missing channels retain last state.
        for name, channel in config.SERVO_CHANNELS.items():
            self._target[name] = normalize_servo(name, channel.get('rest', channel['neutral']))
        self._target['visor'] = 0.
        self._current = dict(self._target)
        self._rig = RigViewState()
        self._rig._background = show_background
        self._rig.pose(self._current)
        self._view = QQuickWidget(self)
        self._view.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        self._view.setClearColor(QColor('#07111a') if show_background else QColor(Qt.GlobalColor.transparent))
        if not show_background:
            self._view.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self._view.rootContext().setContextProperty('rig', self._rig)
        self._view.setSource(QUrl.fromLocalFile(str(ASSET_DIR/'Avatar.qml')))
        if self._view.status() == QQuickWidget.Status.Error:
            raise RuntimeError('3D avatar failed to load: ' + '; '.join(e.toString() for e in self._view.errors()))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self._view)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick_3d)
        self._timer.start()

    def paintEvent(self, event):
        # QQuickWidget owns painting; the inherited class supplies state only.
        pass

    def _tick_3d(self):
        if not self.isVisible():
            return
        self._smooth()
        self._tick_eye_animation()
        self._rig.pose(self._current)
        rgb = _eye_color(self._eye_state)
        brightness = self._eye_brightness() if self._eye_state.get('eyes_active') and self._blink_state != 'closed' else 0.
        self._rig._eye = QColor(*(int(c*brightness) for c in rgb))
        state = chest_render_state(self._chest_state, time.time())
        level = state['brightness'] if state['on'] else 0.
        self._rig._led = lamp_colors(state, time.time())
        pixels = self._rig.mouth_animation.render(self._eye_state, time.time())
        self._rig._mouth = [QColor.fromRgbF(*row) for row in pixels]
        self._rig.changed.emit()


def lamp_colors(state, now):
    colors = []
    step = int(now * state['rate'])
    brightness = state['brightness'] if state['on'] else 0.
    for i in range(9):
        rgb = _pick(state['squares'], _prand(step // 3, i, 5))
        level = brightness * (.7 + .3*_prand(step, i, 99))
        if state['fill'] is not None:
            lit = i < round(9*state['fill'])
            if state['gauge']:
                rgb = _chest_gauge_color(i/8)
                if state['charging'] and i == round(9*state['fill']) + step % max(1,9-round(9*state['fill'])):
                    lit, rgb = True, (30,220,255)
            level = brightness if lit else 0.
        if state['flash'] and brightness:
            rgb = (240,246,255) if (int(now*8)+i)%2 else (80,145,255)
        colors.append(QColor(*(int(max(0,min(255,c*level))) for c in rgb)))
    return colors
