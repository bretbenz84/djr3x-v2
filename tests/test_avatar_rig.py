"""Servo limits, rigid hierarchy and mechanism anchors without hardware or Qt."""
import unittest
import numpy as np
from gui.avatar_rig import (SPEC, LEVEL_PITCH_NORM, pose_matrices, servo_values, point, spring_points,
                            pitch_norm, tilt_norm)

# Level head: head_pitch sits at the rig's level norm (1/3 of the -15..30 travel), tilt straight.
REST = dict(pokerarm=.5,heroarm=.5,elbow=0,hand=.5,neck=.5,headlift=SPEC['lift_reference'],
            head_pitch=LEVEL_PITCH_NORM,head_tilt=.5,visor=1)
class AvatarRigTest(unittest.TestCase):
    def test_physical_ranges_and_head_pitch(self):
        low=servo_values(dict.fromkeys(REST,0));high=servo_values(dict.fromkeys(REST,1))
        for key,span in [('pokerarm',20),('heroarm',20),('neck',45),('elbow',35),('hand',180)]:
            self.assertAlmostEqual(high[key]-low[key],span)
        # norm 0 = fully down, 1 = fully up; the old single-servo angles, re-ordered.
        self.assertEqual((low['head_pitch'],high['head_pitch']),(-15,30))
        self.assertEqual((low['head_tilt'],high['head_tilt']),(-12,12))
        self.assertEqual((low['visor'],high['visor']),(27,0))
        self.assertAlmostEqual(high['headlift']-low['headlift'],.06)
        self.assertAlmostEqual(servo_values(REST)['head_pitch'],0)
        self.assertAlmostEqual(servo_values(REST)['head_tilt'],0)
    def test_pitch_and_tilt_fractions_map_onto_the_rig(self):
        self.assertAlmostEqual(pitch_norm(0),LEVEL_PITCH_NORM)
        self.assertEqual((pitch_norm(-1),pitch_norm(1)),(0,1))
        self.assertAlmostEqual(servo_values(dict(REST,head_pitch=pitch_norm(.5)))['head_pitch'],15)
        self.assertAlmostEqual(servo_values(dict(REST,head_pitch=pitch_norm(-.5)))['head_pitch'],-7.5)
        self.assertEqual((tilt_norm(-1),tilt_norm(0),tilt_norm(1)),(0,.5,1))
        for bad in [float('nan'),float('inf'),None,'x',5,-5]:
            self.assertTrue(0<=pitch_norm(bad)<=1);self.assertTrue(0<=tilt_norm(bad)<=1)
    def test_pitch_up_lifts_the_face_and_tilt_rolls_it_sideways(self):
        face=np.array(SPEC['head_pivot'])+.2*np.array(SPEC['head_forward'])
        level=point(pose_matrices(REST)['head'],face)
        up=point(pose_matrices(dict(REST,head_pitch=1))['head'],face)
        down=point(pose_matrices(dict(REST,head_pitch=0))['head'],face)
        self.assertGreater(up[2],level[2]);self.assertLess(down[2],level[2])
        # A side tilt leaves a point on the forward axis alone and moves the top of the head.
        top=np.array(SPEC['head_pivot'])+[0,0,.2]
        right=pose_matrices(dict(REST,head_tilt=1));left=pose_matrices(dict(REST,head_tilt=0))
        np.testing.assert_allclose(point(right['head'],face),level,atol=1e-8)
        rex_right=np.cross(SPEC['head_forward'],[0,0,1])
        self.assertGreater((point(right['head'],top)-top)@rex_right,0)
        self.assertLess((point(left['head'],top)-top)@rex_right,0)
    def test_rest_and_all_transforms_are_rigid(self):
        for m in pose_matrices(REST).values():np.testing.assert_allclose(m,np.eye(4),atol=1e-8)
        for value in [0,.25,.5,.75,1]:
            matrices=pose_matrices(dict.fromkeys(REST,value))
            np.testing.assert_array_equal(matrices['fixed'],np.eye(4))
            for m in matrices.values():
                np.testing.assert_allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-8)
                self.assertAlmostEqual(np.linalg.det(m[:3,:3]),1)
    def test_hierarchy_keeps_joint_anchors_coincident(self):
        for value in [0,.5,1]:
            m=pose_matrices(dict.fromkeys(REST,value))
            for a,b,p in [('hero','elbow','elbow_pivot'),('elbow','wrist','wrist_pivot'),('neck','head','head_pivot'),('head','visor','visor_pivot')]:
                np.testing.assert_allclose(point(m[a],SPEC[p]),point(m[b],SPEC[p]),atol=1e-8)
            np.testing.assert_allclose(point(m['piston_body'],SPEC['piston_upper']),point(m['elbow'],SPEC['piston_upper']),atol=1e-8)
            np.testing.assert_allclose(point(m['piston_rod'],SPEC['piston_lower']),point(m['hero'],SPEC['piston_lower']),atol=1e-8)
    def test_neck_is_independent_and_spring_extends(self):
        m=pose_matrices(REST);turned=pose_matrices(dict(REST,heroarm=1,pokerarm=0))
        np.testing.assert_allclose(m['head'],turned['head'])
        low=spring_points(pose_matrices(dict(REST,headlift=0)))
        high=spring_points(pose_matrices(dict(REST,headlift=1)))
        np.testing.assert_allclose(low[0],high[0]);self.assertAlmostEqual(high[-1,2]-low[-1,2],.06)
    def test_invalid_and_out_of_range_are_bounded(self):
        for v in [-100,100,float('nan'),float('inf')]:
            for m in pose_matrices(dict.fromkeys(REST,v)).values():self.assertTrue(np.isfinite(m).all())

if __name__=='__main__':unittest.main()
