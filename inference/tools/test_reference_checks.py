import unittest
from reference_checks import reject_resource_failure

class ResourceOracleTests(unittest.TestCase):
    def test_resource_error_is_rethrown_without_turning_it_into_a_golden_failure(self):
        for text in ['CUDA error: out of memory', 'CUDA OUT OF MEMORY', 'CPU allocator: out of memory']:
            error=RuntimeError(text)
            with self.assertRaises(RuntimeError) as raised:
                reject_resource_failure(error)
            self.assertIs(raised.exception,error)

    def test_reference_validation_failure_can_still_be_recorded(self):
        reject_resource_failure(ValueError('K must be divisible by 64'))

if __name__=='__main__':unittest.main()
