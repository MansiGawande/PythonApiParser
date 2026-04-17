-- ============================================================
-- reset_all_and_reparse.sql
-- Run this in SSMS against your ATS database ONCE.
-- Then RESTART the .NET API. The worker will pick up every
-- resume and re-parse them via Python.
-- ============================================================

USE [ATS];   -- change if your DB name differs

-- 1. Reset all queue items (including ones with RetryCount >= 3)
UPDATE [dbo].[ResumeProcessingQueues]
SET    [Status]      = N'Pending',
       [RetryCount]  = 0,
       [ProcessedAt] = NULL;

-- 2. Also enqueue any resumes that never got a queue row
INSERT INTO [dbo].[ResumeProcessingQueues] ([ResumeId], [Status], [RetryCount], [CreatedAt])
SELECT r.[ResumeId], N'Pending', 0, GETDATE()
FROM   [dbo].[Resumes] r
WHERE  NOT EXISTS (
    SELECT 1 FROM [dbo].[ResumeProcessingQueues] q WHERE q.[ResumeId] = r.[ResumeId]
);

-- 3. Reset Parsed flag so updated text is re-extracted
UPDATE [dbo].[Resumes] SET [Parsed] = 0;

-- 4. Wipe old NLP data (so there are no stale rows)
DELETE FROM [dbo].[CandidateSkills];
DELETE FROM [dbo].[CandidateExperiences];
DELETE FROM [dbo].[CandidateEducations];
DELETE FROM [dbo].[ResumeTexts];

-- Verify: should show all rows as Pending / RetryCount = 0
SELECT [QueueId], [ResumeId], [Status], [RetryCount], [ProcessedAt]
FROM   [dbo].[ResumeProcessingQueues]
ORDER BY [QueueId];
