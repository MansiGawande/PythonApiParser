-- ============================================================
-- Migration: addResumeIdJobApplication
-- Run this ONCE in SQL Server Management Studio (SSMS)
-- against your ATS database.
-- All statements are idempotent (safe to run even if partially applied).
-- ============================================================

-- 1. Add ResumeId column (if missing)
IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID(N'dbo.JobApplications')
      AND name = N'ResumeId'
)
BEGIN
    ALTER TABLE [dbo].[JobApplications] ADD [ResumeId] INT NULL;
    PRINT 'Added column ResumeId to JobApplications.';
END
ELSE
BEGIN
    PRINT 'Column ResumeId already exists – skipping.';
END
GO

-- 2. Create index on ResumeId (if missing)
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.JobApplications')
      AND name = N'IX_JobApplications_ResumeId'
)
BEGIN
    CREATE INDEX [IX_JobApplications_ResumeId]
        ON [dbo].[JobApplications] ([ResumeId]);
    PRINT 'Created index IX_JobApplications_ResumeId.';
END
ELSE
BEGIN
    PRINT 'Index IX_JobApplications_ResumeId already exists – skipping.';
END
GO

-- 3. Add unique index on (JobId, CandidateId) — one application per job per candidate
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.JobApplications')
      AND name = N'IX_JobApplications_JobId_CandidateId'
)
BEGIN
    CREATE UNIQUE INDEX [IX_JobApplications_JobId_CandidateId]
        ON [dbo].[JobApplications] ([JobId], [CandidateId]);
    PRINT 'Created unique index IX_JobApplications_JobId_CandidateId.';
END
ELSE
BEGIN
    PRINT 'Unique index IX_JobApplications_JobId_CandidateId already exists – skipping.';
END
GO

-- 4. Add FK to Resumes table (if missing)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_JobApplications_Resumes_ResumeId'
)
BEGIN
    ALTER TABLE [dbo].[JobApplications]
        ADD CONSTRAINT [FK_JobApplications_Resumes_ResumeId]
        FOREIGN KEY ([ResumeId])
        REFERENCES [dbo].[Resumes] ([ResumeId])
        ON DELETE SET NULL;
    PRINT 'Added FK_JobApplications_Resumes_ResumeId.';
END
ELSE
BEGIN
    PRINT 'FK_JobApplications_Resumes_ResumeId already exists – skipping.';
END
GO

-- 5. Record migration in EF history table (so dotnet ef knows it is applied)
IF NOT EXISTS (
    SELECT 1 FROM [dbo].[__EFMigrationsHistory]
    WHERE [MigrationId] = N'20260330085651_addResumeIdJobApplication'
)
BEGIN
    INSERT INTO [dbo].[__EFMigrationsHistory] ([MigrationId], [ProductVersion])
    VALUES (N'20260330085651_addResumeIdJobApplication', N'9.0.3');
    PRINT 'Recorded migration in __EFMigrationsHistory.';
END
ELSE
BEGIN
    PRINT 'Migration already recorded in __EFMigrationsHistory – skipping.';
END
GO

PRINT '=== Migration complete ===';
