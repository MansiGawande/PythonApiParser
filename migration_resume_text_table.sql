-- ResumeTexts: stores plain text extracted by the Python parser (one row per resume).
-- Run if you apply schema manually instead of dotnet ef database update.

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = N'ResumeTexts')
BEGIN
    CREATE TABLE [dbo].[ResumeTexts] (
        [Id] INT IDENTITY(1,1) NOT NULL,
        [ResumeId] INT NOT NULL,
        [ExtractedText] NVARCHAR(MAX) NOT NULL,
        [CreatedAt] DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT [PK_ResumeTexts] PRIMARY KEY ([Id]),
        CONSTRAINT [FK_ResumeTexts_Resumes_ResumeId] FOREIGN KEY ([ResumeId])
            REFERENCES [dbo].[Resumes] ([ResumeId]) ON DELETE CASCADE
    );
    CREATE UNIQUE INDEX [IX_ResumeTexts_ResumeId] ON [dbo].[ResumeTexts] ([ResumeId]);
END
GO

IF NOT EXISTS (SELECT 1 FROM [dbo].[__EFMigrationsHistory] WHERE [MigrationId] = N'20260331120000_AddResumeTextTable')
BEGIN
    INSERT INTO [dbo].[__EFMigrationsHistory] ([MigrationId], [ProductVersion])
    VALUES (N'20260331120000_AddResumeTextTable', N'8.0.2');
END
GO
