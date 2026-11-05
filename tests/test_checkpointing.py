import os
from pathlib import Path
import tempfile
import unittest
from noposplat_jittor.checkpointing import save_checkpoint

TEMP_ROOT=Path(os.environ.get('NPS_TEST_TMP',str(Path(__file__).resolve().parents[1]/'.test_tmp')))
TEMP_ROOT.mkdir(exist_ok=True)


class CheckpointPublication(unittest.TestCase):
    def test_completed_file_is_published(self):
        with tempfile.TemporaryDirectory(prefix='nps-checkpoint-',dir=TEMP_ROOT) as folder:
            root=Path(folder).resolve();self.assertTrue(root.is_relative_to(TEMP_ROOT.resolve()))
            destination=root/'step_000001.pkl'
            save_checkpoint({},destination,writer=lambda state,path:Path(path).write_bytes(b'complete'))
            self.assertEqual(destination.read_bytes(),b'complete');self.assertEqual(list(root.iterdir()),[destination])

    def test_failed_dump_never_appears_as_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='nps-checkpoint-',dir=TEMP_ROOT) as folder:
            root=Path(folder).resolve();self.assertTrue(root.is_relative_to(TEMP_ROOT.resolve()))
            destination=root/'step_000001.pkl'
            def interrupted(state,path):Path(path).write_bytes(b'partial');raise OSError('simulated interrupted save')
            with self.assertRaises(OSError):save_checkpoint({},destination,writer=interrupted)
            self.assertFalse(destination.exists());self.assertEqual(list(root.iterdir()),[])

    def test_existing_checkpoint_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix='nps-checkpoint-',dir=TEMP_ROOT) as folder:
            root=Path(folder).resolve();self.assertTrue(root.is_relative_to(TEMP_ROOT.resolve()))
            destination=root/'step_000001.pkl';destination.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):save_checkpoint({},destination,writer=lambda *unused:self.fail('Must not invoke writer'))
            self.assertEqual(destination.read_bytes(),b'existing')


class CheckpointRetention(unittest.TestCase):
    def test_only_registered_published_files_can_be_removed(self):
        from noposplat_jittor.checkpointing import retain_owned_checkpoints
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as d:
            root=Path(d);unrelated=root/'step_000000.pkl';unrelated.write_bytes(b'user checkpoint')
            for i in range(1,4):
                file=root/f'step_{i:06d}.pkl';file.write_bytes(bytes([i])*20);retain_owned_checkpoints(file,2)
            self.assertFalse((root/'step_000001.pkl').exists());self.assertTrue((root/'step_000002.pkl').exists());self.assertTrue((root/'step_000003.pkl').exists())
            self.assertEqual(unrelated.read_bytes(),b'user checkpoint')

    def test_changed_registered_file_is_preserved_and_rejected(self):
        from noposplat_jittor.checkpointing import retain_owned_checkpoints
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as d:
            root=Path(d)
            for i in (1,2):
                file=root/f'step_{i:06d}.pkl';file.write_bytes(bytes([i]));retain_owned_checkpoints(file,2)
            (root/'step_000001.pkl').write_bytes(b'changed by user');last=root/'step_000003.pkl';last.write_bytes(b'new checkpoint')
            with self.assertRaises(ValueError):retain_owned_checkpoints(last,2)
            self.assertEqual((root/'step_000001.pkl').read_bytes(),b'changed by user');self.assertTrue(last.exists())


if __name__=='__main__':unittest.main()
