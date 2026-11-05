"""Original bounded preparation with explicit ZIP integrity candidate accounting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tools.prepare_reduced_experiments as original
from tools.checked_candidate_conversion import CheckedCandidateConversion
original.ChunkConversion=CheckedCandidateConversion
release=original.release_verified_source
def release_good_only(source,row,conversion,cache):
    record=conversion.state['chunks'][row['path']]
    if record.get('preserved_corrupt_source'):return
    return release(source,row,conversion,cache)
original.release_verified_source=release_good_only
if __name__=='__main__':original.main()
