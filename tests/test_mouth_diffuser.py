"""Verify scattering preserves off/color state and the exported LED ordering."""
import json
import unittest
import numpy as np
from gui.avatar_rig import ASSET_DIR
from gui.rex_avatar_3d import MouthDiffuser, _geometry_data

class MouthDiffuserTests(unittest.TestCase):
    def setUp(self):
        batches = json.loads((ASSET_DIR / 'materials.json').read_text())
        mouth = sorted((b for b in batches if b['kind'] == 'mouth'), key=lambda b: b['lamp'])
        self.diffuser = MouthDiffuser([_geometry_data()[b['id']] for b in mouth])

    def pixels(self):
        return np.frombuffer(bytes(self.diffuser.textureData()), dtype=np.uint8).reshape(64,64,4)

    def test_off_is_not_emissive(self):
        self.assertFalse(self.pixels()[:,:,:3].any())
        self.assertTrue((self.pixels()[:,:,3] == 255).all())

    def test_uniform_color_stays_uniform_without_white_leak(self):
        self.diffuser.render(np.tile([0.,.2,0.], (80,1)))
        rgb = self.pixels()[:,:,:3]
        self.assertFalse(rgb[:,:,0].any())
        self.assertFalse(rgb[:,:,2].any())
        self.assertEqual(np.unique(rgb[:,:,1]).size, 1)

    def test_single_led_spreads_locally_and_preserves_array_orientation(self):
        colors = np.zeros((80,3)); colors[0,2] = .5
        self.diffuser.render(colors)
        blue = self.pixels()[:,:,2]
        y,x = np.unravel_index(blue.argmax(),blue.shape)
        self.assertLess(x,16)  # first exported column
        self.assertGreater(y,47)  # first exported row is at the top in model UVs
        self.assertGreater(np.count_nonzero(blue),30)
        self.assertGreater(np.unique(blue).size,10)  # smooth falloff, not binary LEDs
        self.assertEqual(blue[0,-1],0)

if __name__ == '__main__':
    unittest.main()
