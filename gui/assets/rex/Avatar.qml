import QtQuick
import QtQuick3D
import QtQuick3D.Effects

Item {
    id: root
    Rectangle {
        anchors.fill: parent
        visible: rig && rig.background
        gradient: Gradient {
            GradientStop { position: 0; color: "#323b3c" }
            GradientStop { position: 0.55; color: "#697778" }
            GradientStop { position: 1; color: "#b0b4ac" }
        }
    }
    Repeater {
        model: rig && rig.background ? 40 : 0
        delegate: Rectangle {
            required property int index
            width: Math.min(root.width * 0.58, root.height * 0.58) * (1 - index * 0.011)
            height: width * 0.12
            x: root.width * 0.5 - width * 0.5
            y: root.height * 0.94 - height * 0.5
            radius: width
            color: "#1d282b"
            opacity: 0.010
        }
    }
    View3D {
        anchors.fill: parent
        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Transparent
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
            lightProbe: Texture { source: "studio.hdr" }
            probeExposure: 0.7
            effects: [HDRBloomTonemap { bloomThreshold: 1.5; blurFalloff: 4; tonemappingLerp: 0 }]
        }
        Node {
            position: Qt.vector3d(5, 51, 0)
            eulerRotation: Qt.vector3d(-8, 25, 0)
            OrthographicCamera {
                z: 250
                clipNear: 1
                clipFar: 1000
                horizontalMagnification: Math.max(0.01, Math.min(root.width / 100, root.height / 112))
                verticalMagnification: horizontalMagnification
            }
        }
        DirectionalLight { eulerRotation: Qt.vector3d(-35, -30, 0); brightness: 1.8; ambientColor: '#919ca3'; castsShadow: true; shadowFactor: 45 }
        DirectionalLight { eulerRotation: Qt.vector3d(-20, 140, 0); brightness: 1.0 }
        DirectionalLight { eulerRotation: Qt.vector3d(30, 70, 0); brightness: 0.6 }
        Node {
            eulerRotation.x: -90
            scale: Qt.vector3d(100, 100, 100)
            Repeater3D {
                model: rig ? rig.batches : []
                delegate: Model {
                    required property var modelData
                    geometry: modelData.geometry
                    position: rig ? rig.transforms[modelData.group].position : Qt.vector3d(0,0,0)
                    rotation: rig ? rig.transforms[modelData.group].rotation : Qt.quaternion(1,0,0,0)
                    materials: PrincipledMaterial {
                        baseColorMap: modelData.kind === 'visor' ? visorTexture : null
                        Texture { id: visorTexture; source: "visor_stripes.svg" }
                        baseColor: !rig ? modelData.color : modelData.kind === 'visor' ? "white" : modelData.kind === 'mouth' ? rig.mouthColors[modelData.lamp] : modelData.kind === 'eye' ? rig.eyeColor : modelData.kind === 'led' ? rig.ledColors[modelData.lamp] : modelData.color
                        metalness: (modelData.kind === 'mouth' || modelData.kind === 'eye') ? 0 : modelData.metalness * 0.8
                        roughness: Math.max(modelData.group === "neck" ? 0.18 : 0.35, modelData.roughness)
                        transmissionFactor: (modelData.kind === 'mouth' || modelData.kind === 'eye') ? 0.15 : 0
                        cullMode: Material.NoCulling
                        emissiveFactor: !rig ? Qt.vector3d(0,0,0) : modelData.kind === 'mouth' ? Qt.vector3d(rig.mouthColors[modelData.lamp].r * 5, rig.mouthColors[modelData.lamp].g * 5, rig.mouthColors[modelData.lamp].b * 5) : modelData.kind === 'eye' ? Qt.vector3d(rig.eyeColor.r * 3, rig.eyeColor.g * 3, rig.eyeColor.b * 3) : modelData.kind === 'led' ? Qt.vector3d(rig.ledColors[modelData.lamp].r, rig.ledColors[modelData.lamp].g, rig.ledColors[modelData.lamp].b) : Qt.vector3d(0,0,0)
                    }
                }
            }
        }
    }
}
