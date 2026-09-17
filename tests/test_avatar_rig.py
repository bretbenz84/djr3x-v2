"""Servo limits, rigid hierarchy and mechanism anchors without hardware or Qt."""
import unittest
import numpy as np
from gui.avatar_rig import SPEC, pose_matrices, servo_values, point, spring_points

REST = dict(pokerarm=.5,heroarm=.5,elbow=0,hand=.5,neck=.5,headlift=SPEC['lift_reference'],headtilt=2/3,visor=1)
class AvatarRigTest(unittest.TestCase):
    def test_physical_ranges_and_inverted_tilt(self):
        low=servo_values(dict.fromkeys(REST,0));high=servo_values(dict.fromkeys(REST,1))
        for key,span in [('pokerarm',20),('heroarm',20),('neck',45),('elbow',35),('hand',180)]:
            self.assertAlmostEqual(high[key]-low[key],span)
        self.assertEqual((low['headtilt'],high['headtilt']),(30,-15))
        self.assertEqual((low['visor'],high['visor']),(27,0))
        self.assertAlmostEqual(high['headlift']-low['headlift'],.06)
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
