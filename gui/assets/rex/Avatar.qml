import QtQuick
import QtQuick3D
import QtQuick3D.Effects

Item {
    id: root
    Rectangle {
        anchors.fill: parent
        visible: rig && rig.background
        gradient: Gradient {
            GradientStop { position: 0; color: "#252e29" }
            GradientStop { position: 0.55; color: "#56635c" }
            GradientStop { position: 1; color: "#778074" }
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
        // The bay uses the same camera and ground plane as the droid so the
        // deck and contact shadow remain attached as the window is resized.
        Node {
            visible: rig && rig.background
            PrincipledMaterial { id: structure; baseColor: "#35443c"; metalness: 0.55; roughness: 0.7 }
            PrincipledMaterial { id: deck; baseColor: "#4b5549"; metalness: 0.45; roughness: 0.8 }
            PrincipledMaterial { id: trim; baseColor: "#bb8750"; metalness: 0.6; roughness: 0.6 }
            PrincipledMaterial { id: lamp; baseColor: "#ffe6b4"; emissiveFactor: Qt.vector3d(1.2, 0.65, 0.25) }
            Model { source: "#Cube"; position: Qt.vector3d(0,-6,0); scale: Qt.vector3d(5,0.06,5); materials: deck }
            Model { source: "#Cylinder"; position: Qt.vector3d(0,-3,0); scale: Qt.vector3d(0.96,0.025,0.96); materials: structure }
            Model { source: "#Cylinder"; position: Qt.vector3d(0,-1.7,0); scale: Qt.vector3d(0.90,0.008,0.90); materials: trim }
            Model { source: "#Cylinder"; position: Qt.vector3d(0,-0.8,0); scale: Qt.vector3d(0.86,0.016,0.86); materials: deck }
            Node {
                x: 5
                eulerRotation.y: 25
                Model {
                    source: "#Cube"
                    position: Qt.vector3d(0, 55, -66)
                    scale: Qt.vector3d(2.6, 1.7, 0.04)
                    materials: PrincipledMaterial {
                        baseColorMap: Texture { source: "bay_wall.svg" }
                        baseColor: "#7a8170"
                        roughness: 0.9
                        metalness: 0.2
                    }
                }
                Repeater3D {
                    model: [-44, 44]
                    delegate: Node {
                        required property real modelData
                        x: modelData
                        Model { source: "#Cube"; position: Qt.vector3d(0,58,-49); scale: Qt.vector3d(0.06,1.4,0.12); materials: structure }
                        Model { source: "#Cube"; position: Qt.vector3d(0,57,-42); scale: Qt.vector3d(0.011,0.66,0.012); materials: lamp }
                    }
                }
                Model { source: "#Cube"; position: Qt.vector3d(0,125,-49); scale: Qt.vector3d(1.25,0.06,0.12); materials: structure }
            }
            Repeater3D {
                model: 12
                delegate: Model {
                    required property int index
                    source: "#Cube"
                    position: Qt.vector3d(-37 + index * 6.5, -2.8, 40)
                    scale: Qt.vector3d(0.032,0.008,0.10)
                    eulerRotation.y: -30
                    materials: trim
                }
            }
        }
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
    Item {
        anchors.fill: parent
        visible: rig && rig.background
        Text {
            anchors { top: parent.top; left: parent.left; margins: 18 }
            text: "R3X  /  SERVICE BAY"
            color: "#dbd6bf"; font.pixelSize: 10; font.letterSpacing: 2
        }
        Text {
            anchors { bottom: parent.bottom; left: parent.left; margins: 18 }
            text: "BLACK SPIRE OUTPOST     //     03"
            color: "#c0c9b8"; font.pixelSize: 9; font.letterSpacing: 1.5
        }
    }
}
