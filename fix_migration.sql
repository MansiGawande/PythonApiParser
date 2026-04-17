-- ============================================================
-- ATS Database Migration Fix Script
-- Run this in SQL Server Management Studio (SSMS) against your ATS database
-- ============================================================

USE [ATS];
GO

-- -------------------------------------------------------
-- 1. Add OriginalFileName column to Resumes (if missing)
-- -------------------------------------------------------
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'Resumes' AND COLUMN_NAME = 'OriginalFileName'
)
BEGIN
    ALTER TABLE [dbo].[Resumes]
        ADD [OriginalFileName] nvarchar(255) NULL;
    PRINT 'Added OriginalFileName column to Resumes.';
END
ELSE
    PRINT 'OriginalFileName already exists - skipped.';
GO

-- -------------------------------------------------------
-- 2. Add IsActive column to Resumes (if missing)
-- -------------------------------------------------------
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'Resumes' AND COLUMN_NAME = 'IsActive'
)
BEGIN
    ALTER TABLE [dbo].[Resumes]
        ADD [IsActive] bit NOT NULL CONSTRAINT [DF_Resumes_IsActive] DEFAULT (1);
    -- Set all existing resumes to active
    UPDATE [dbo].[Resumes] SET [IsActive] = 1;
    PRINT 'Added IsActive column to Resumes and set all existing rows to active.';
END
ELSE
    PRINT 'IsActive already exists - skipped.';
GO

-- -------------------------------------------------------
-- 3. Create ResumeTexts table (if missing)
-- -------------------------------------------------------
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'ResumeTexts'
)
BEGIN
    CREATE TABLE [dbo].[ResumeTexts] (
        [Id]            int            NOT NULL IDENTITY(1,1),
        [ResumeId]      int            NOT NULL,
        [ExtractedText] nvarchar(max)  NOT NULL,
        [CreatedAt]     datetime2      NOT NULL CONSTRAINT [DF_ResumeTexts_CreatedAt] DEFAULT (GETDATE()),

        CONSTRAINT [PK_ResumeTexts] PRIMARY KEY CLUSTERED ([Id]),
        CONSTRAINT [FK_ResumeTexts_Resumes_ResumeId]
            FOREIGN KEY ([ResumeId]) REFERENCES [dbo].[Resumes]([ResumeId])
            ON DELETE CASCADE
    );
    CREATE UNIQUE INDEX [IX_ResumeTexts_ResumeId] ON [dbo].[ResumeTexts] ([ResumeId]);
    PRINT 'Created ResumeTexts table.';
END
ELSE
    PRINT 'ResumeTexts already exists - skipped.';
GO

-- -------------------------------------------------------
-- 4. Create ResumeProcessingQueues table (if missing)
-- -------------------------------------------------------
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'ResumeProcessingQueues'
)
BEGIN
    CREATE TABLE [dbo].[ResumeProcessingQueues] (
        [QueueId]     int          NOT NULL IDENTITY(1,1),
        [ResumeId]    int          NOT NULL,
        [Status]      nvarchar(20) NOT NULL CONSTRAINT [DF_RPQ_Status] DEFAULT (N'Pending'),
        [RetryCount]  int          NOT NULL CONSTRAINT [DF_RPQ_RetryCount] DEFAULT (0),
        [CreatedAt]   datetime2    NOT NULL CONSTRAINT [DF_RPQ_CreatedAt] DEFAULT (GETDATE()),
        [ProcessedAt] datetime2    NULL,

        CONSTRAINT [PK_ResumeProcessingQueues] PRIMARY KEY CLUSTERED ([QueueId]),
        CONSTRAINT [FK_RPQ_Resumes_ResumeId]
            FOREIGN KEY ([ResumeId]) REFERENCES [dbo].[Resumes]([ResumeId])
            ON DELETE CASCADE
    );
    CREATE INDEX [IX_RPQ_Status]   ON [dbo].[ResumeProcessingQueues] ([Status]);
    CREATE INDEX [IX_RPQ_ResumeId] ON [dbo].[ResumeProcessingQueues] ([ResumeId]);
    PRINT 'Created ResumeProcessingQueues table.';
END
ELSE
    PRINT 'ResumeProcessingQueues already exists - skipped.';
GO

-- -------------------------------------------------------
-- 5. Queue existing un-parsed resumes for processing
--    (only if they don't already have a queue entry)
-- -------------------------------------------------------
INSERT INTO [dbo].[ResumeProcessingQueues] ([ResumeId], [Status], [RetryCount], [CreatedAt])
SELECT r.[ResumeId], N'Pending', 0, GETDATE()
FROM   [dbo].[Resumes] r
WHERE  r.[Parsed] = 0
  AND  NOT EXISTS (
           SELECT 1 FROM [dbo].[ResumeProcessingQueues] q
           WHERE  q.[ResumeId] = r.[ResumeId]
       );

PRINT CAST(@@ROWCOUNT AS varchar) + ' existing un-parsed resumes queued for processing.';
GO

-- -------------------------------------------------------
-- 6. Register migrations in EF history so EF does NOT
--    try to re-run them when the app starts
-- -------------------------------------------------------
IF NOT EXISTS (
    SELECT 1 FROM [dbo].[__EFMigrationsHistory]
    WHERE [MigrationId] = '20260331083824_AddFileOriginalName'
)
BEGIN
    INSERT INTO [dbo].[__EFMigrationsHistory] ([MigrationId], [ProductVersion])
    VALUES ('20260331083824_AddFileOriginalName', '8.0.2');
    PRINT 'Registered migration 20260331083824_AddFileOriginalName.';
END
GO

IF NOT EXISTS (
    SELECT 1 FROM [dbo].[__EFMigrationsHistory]
    WHERE [MigrationId] = '20260331120000_AddResumeTextTable'
)
BEGIN
    INSERT INTO [dbo].[__EFMigrationsHistory] ([MigrationId], [ProductVersion])
    VALUES ('20260331120000_AddResumeTextTable', '8.0.2');
    PRINT 'Registered migration 20260331120000_AddResumeTextTable.';
END
GO

-- -------------------------------------------------------
-- 7. Verify final schema
-- -------------------------------------------------------
SELECT 'Resumes columns' AS [Check],
       COLUMN_NAME, DATA_TYPE, IS_NULLABLE
FROM   INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_NAME = 'Resumes'
ORDER  BY ORDINAL_POSITION;

SELECT 'ResumeTexts exists' AS [Check], COUNT(*) AS [Rows] FROM [dbo].[ResumeTexts];
SELECT 'ResumeProcessingQueues' AS [Check], COUNT(*) AS [Rows] FROM [dbo].[ResumeProcessingQueues];
SELECT 'EF Migrations' AS [Check], [MigrationId] FROM [dbo].[__EFMigrationsHistory] ORDER BY 2;
GO
