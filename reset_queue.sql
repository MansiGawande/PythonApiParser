-- ============================================================
-- ATS – Reset Failed Queue Items so they are retried
-- Run in SSMS against your ATS database whenever you need to
-- force a re-parse (e.g. after starting the Python service).
-- ============================================================

USE [ATS];
GO

-- ── 1. Show current queue status (before reset) ──────────────────
SELECT
    q.QueueId,
    q.ResumeId,
    q.Status,
    q.RetryCount,
    q.CreatedAt,
    q.ProcessedAt,
    r.FilePath,
    r.OriginalFileName,
    r.Parsed
FROM [dbo].[ResumeProcessingQueues] q
JOIN [dbo].[Resumes] r ON r.ResumeId = q.ResumeId
ORDER BY q.QueueId;
GO

-- ── 2. Reset ALL Failed items back to Pending ────────────────────
--    (sets RetryCount = 0 so the worker tries again from scratch)
UPDATE [dbo].[ResumeProcessingQueues]
SET
    [Status]      = 'Pending',
    [RetryCount]  = 0,
    [ProcessedAt] = NULL
WHERE [Status] IN ('Failed', 'Processing');     -- also reset any stuck "Processing"

PRINT CAST(@@ROWCOUNT AS varchar) + ' queue item(s) reset to Pending.';
GO

-- ── 3. Also un-mark Parsed = 0 for those resumes ─────────────────
--    so the Parsed column is accurate after re-processing
UPDATE r
SET r.[Parsed] = 0
FROM [dbo].[Resumes] r
JOIN [dbo].[ResumeProcessingQueues] q ON q.ResumeId = r.ResumeId
WHERE q.Status = 'Pending';

PRINT CAST(@@ROWCOUNT AS varchar) + ' resume(s) marked Parsed = 0 for re-processing.';
GO

-- ── 4. Verify (after reset) ───────────────────────────────────────
SELECT
    q.QueueId,
    q.ResumeId,
    q.Status,
    q.RetryCount,
    r.OriginalFileName,
    r.Parsed
FROM [dbo].[ResumeProcessingQueues] q
JOIN [dbo].[Resumes] r ON r.ResumeId = q.ResumeId
ORDER BY q.QueueId;
GO
