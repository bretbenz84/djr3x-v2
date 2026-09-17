import QtQuick
import QtQuick3D

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
        camera: portraitCamera
        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Transparent
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
            lightProbe: Texture { textureData: rig ? rig.lightProbe : null }
            probeExposure: 0.9
            tonemapMode: SceneEnvironment.TonemapModeAces
            aoEnabled: true
            aoDither: false
            aoSampleRate: 4
            specularAAEnabled: true
            aoStrength: 45
            aoDistance: 8
            aoSoftness: 30
            aoBias: 0.03
            // Use one tone-mapping pass; the legacy bloom effect lifted shadows.
        }
        Node {
            position: Qt.vector3d(5, 51, 0)
            eulerRotation: Qt.vector3d(-8, 25, 0)
            OrthographicCamera {
                id: portraitCamera
                z: 250
                clipNear: 1
                clipFar: 1000
                horizontalMagnification: Math.max(0.01, Math.min(root.width / 100, root.height / 112))
                verticalMagnification: horizontalMagnification
            }
        }
        // Warm window light, restrained cool bounce, and a rear edge highlight.
        PointLight {
            position: Qt.vector3d(-85, 145, 110)
            constantFade: 1; linearFade: 0; quadraticFade: 0.0001
            color: "#fff0d8"; brightness: 4.0; ambientColor: "#000000"
            castsShadow: true; shadowFactor: 75
            shadowMapQuality: Light.ShadowMapQualityHigh
            shadowFilter: 3; shadowBias: 0.03
        }
        PointLight { position: Qt.vector3d(100, 85, 80); color: "#c4ddf4"; brightness: 0.8; constantFade: 1; linearFade: 0; quadraticFade: 0.0001 }
        DirectionalLight { eulerRotation: Qt.vector3d(-25, 155, 0); color: "#d5e9ff"; brightness: 0.18 }
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
                        baseColor: "#465045"
                        roughness: 0.95
                        metalness: 0.0
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
        Texture {
            id: resinEmission
            textureData: rig ? rig.mouthTexture : null
            tilingModeHorizontal: Texture.ClampToEdge
            tilingModeVertical: Texture.ClampToEdge
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
                        baseColor: !rig ? modelData.color : modelData.kind === 'visor' ? "white" : modelData.kind === 'eye' ? rig.eyeColor : modelData.kind === 'led' ? rig.ledColors[modelData.lamp] : modelData.color
                        metalness: (modelData.kind === 'mouth' || modelData.kind === 'eye') ? 0 : modelData.metalness
                        roughness: Math.max(modelData.group === "neck" ? 0.18 : 0.35, modelData.roughness)
                        emissiveMap: modelData.kind === 'mouth' ? resinEmission : null
                        transmissionFactor: modelData.kind === 'eye' ? 0.15 : 0
                        cullMode: Material.NoCulling
                        emissiveFactor: !rig ? Qt.vector3d(0,0,0) : modelData.kind === 'mouth' ? Qt.vector3d(3.5, 3.5, 3.5) : modelData.kind === 'eye' ? Qt.vector3d(rig.eyeColor.r * 3, rig.eyeColor.g * 3, rig.eyeColor.b * 3) : modelData.kind === 'led' ? Qt.vector3d(rig.ledColors[modelData.lamp].r, rig.ledColors[modelData.lamp].g, rig.ledColors[modelData.lamp].b) : Qt.vector3d(0,0,0)
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
