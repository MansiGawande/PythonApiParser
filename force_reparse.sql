-- ============================================================
-- ATS – Force Re-Parse ALL Resumes (even RetryCount >= 3)
-- Run this in SSMS whenever you want to re-trigger NLP parsing.
-- ============================================================

USE [ATS];
GO

-- ── Step 1: Show current queue status ────────────────────────────────
SELECT
    q.QueueId,
    q.ResumeId,
    q.Status,
    q.RetryCount,
    q.ProcessedAt,
    r.OriginalFileName,
    r.FilePath,
    r.Parsed
FROM [dbo].[ResumeProcessingQueues] q
JOIN [dbo].[Resumes] r ON r.ResumeId = q.ResumeId
ORDER BY q.QueueId;
GO

-- ── Step 2: Reset ALL queue items (including RetryCount >= 3) ────────
UPDATE [dbo].[ResumeProcessingQueues]
SET
    [Status]      = 'Pending',
    [RetryCount]  = 0,
    [ProcessedAt] = NULL
WHERE [Status] IN ('Failed', 'Processing', 'Completed');

PRINT CAST(@@ROWCOUNT AS varchar) + ' queue item(s) reset to Pending.';
GO

-- ── Step 3: Reset Parsed flag on ALL resumes ─────────────────────────
UPDATE [dbo].[Resumes]
SET [Parsed] = 0;
PRINT CAST(@@ROWCOUNT AS varchar) + ' resume(s) Parsed flag reset to 0.';
GO

-- ── Step 4: Clear old NLP data so it is re-inserted cleanly ──────────
DELETE FROM [dbo].[CandidateSkills];
PRINT CAST(@@ROWCOUNT AS varchar) + ' CandidateSkills rows deleted.';

DELETE FROM [dbo].[CandidateExperiences];
PRINT CAST(@@ROWCOUNT AS varchar) + ' CandidateExperiences rows deleted.';

DELETE FROM [dbo].[CandidateEducations];
PRINT CAST(@@ROWCOUNT AS varchar) + ' CandidateEducations rows deleted.';

DELETE FROM [dbo].[ResumeTexts];
PRINT CAST(@@ROWCOUNT AS varchar) + ' ResumeTexts rows deleted.';
GO

-- ── Step 5: Add queue entries for any resumes that have no queue entry ─
INSERT INTO [dbo].[ResumeProcessingQueues] ([ResumeId], [Status], [RetryCount], [CreatedAt])
SELECT r.[ResumeId], 'Pending', 0, GETDATE()
FROM   [dbo].[Resumes] r
WHERE  NOT EXISTS (
    SELECT 1 FROM [dbo].[ResumeProcessingQueues] q WHERE q.[ResumeId] = r.[ResumeId]
);
PRINT CAST(@@ROWCOUNT AS varchar) + ' missing queue entries added.';
GO

-- ── Step 6: Verify queue is ready ────────────────────────────────────
SELECT
    q.QueueId,
    q.Status,
    q.RetryCount,
    r.OriginalFileName
FROM [dbo].[ResumeProcessingQueues] q
JOIN [dbo].[Resumes] r ON r.ResumeId = q.ResumeId
ORDER BY q.QueueId;
GO
